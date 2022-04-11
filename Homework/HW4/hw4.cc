#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <list>
#include <set>
#include <mpi.h>
#include <pthread.h>
#include <chrono>
#include <ctime>

#define REQUEST 0
#define DISPATCH 1 
#define REQUEST_REDUCE 2
#define DISPATCH_REDUCE 3

typedef std::pair<std::string, int> Item;
typedef std::pair<int, int> TaskInfo;
// typedef std::pair<int, std::string> Record;


struct mapPool{
    pthread_mutex_t lock;
    pthread_cond_t notify;
    pthread_t *threads;
    std::queue<int> *task;
    std::string infilename;
    int chunk_size;
    int thread_cnt;
    int free_thread_cnt;
    // int head;
    // int tail;
    int shutdown;
    int started;
    int num_reducer;    
};


void *mapperExecute(void * Pool){
    mapPool *pool = (mapPool*)Pool;
    int dataChunk;
    bool isEmpty = false;
    bool init_state = true;

    while(!isEmpty){
        pthread_mutex_lock(&(pool->lock));
        if((*pool->task).empty()){
            if(!init_state) isEmpty = true;
            else {
                pthread_mutex_unlock(&(pool->lock));
                continue;
            }
            // continue;
        }
        else{
            init_state = false;
            dataChunk = (*pool->task).front();

            (*pool->task).pop();
            pool->free_thread_cnt--;
        }
        pthread_mutex_unlock(&(pool->lock));
        if(!isEmpty){
            // Input Split function
            std::ifstream input_file(pool->infilename);
            std::string line;
            std::map<std::string, int> word_count;
            std::string w;
            std::vector<std::string> words;
            std::set<std::string> wordsSet;
            for(int i = 0; i < (dataChunk-1)*pool->chunk_size; i++){
                getline(input_file, line);
            }
            for(int i = 0; i < pool->chunk_size; i++){
                
                getline(input_file, line);
                size_t pos = 0;
                while ((pos = line.find(" ")) != std::string::npos){
                    w = line.substr(0, pos);
                    words.push_back(w);

                    line.erase(0, pos + 1);
                }
                if (!line.empty())
                    words.push_back(line);
            }
            input_file.close();
            
            // Map function
            // opt
            for (auto word : words){
                wordsSet.insert(word);
                if (word_count.count(word) == 0){
                    word_count[word] = 1;
                }else{
                    word_count[word]++;
                }
            }

            // Partition function
            // std::vector<std::set<std::string>> partition_list(pool->num_reducer+1);
            // for(auto word : wordsSet) {
            //     // std::cout << word << "\n";
            //     int reducerID = (int(word[0]) % (pool->num_reducer)) + 1;
            //     partition_list[reducerID].insert(word);
            // }
            // for(int i = 1; i <= pool->num_reducer; i++){
                // std::string num_reducer_s = std::to_string(i);
            std::string dataChunk_s = std::to_string(dataChunk);
            std::ofstream chunk_file("./wordInChunk/chunk" + dataChunk_s + ".out");
            for(auto word : wordsSet){
                chunk_file << word << ' ' << word_count[word] << "\n";
            }
            chunk_file.close();
            // }
        }
        
        pthread_mutex_lock(&(pool->lock));
        pool->free_thread_cnt++;
        pthread_mutex_unlock(&(pool->lock)); 
    }
    pthread_exit(NULL);
}


// Mapper thread pool
void threadpool_init(int rank, int num_thread, std::string input_filename, int chunk_size, int num_reducer){
    // mapPool *pool = (mapPool *)malloc(sizeof(mapPool));
    mapPool *pool = new mapPool;
    pool->infilename = input_filename;
    pool->threads = (pthread_t *)malloc(sizeof(pthread_t) * num_thread);
    pool->task = new std::queue<int>;
    pool->chunk_size = chunk_size;
    // pool->head = pool->tail = pool->thread_cnt = pool->count = 0;
    // pool->shutdown = pool->started = 0;
    pool->thread_cnt = pool->free_thread_cnt = 0; 
    pool->num_reducer = num_reducer;
    
    
    pthread_mutex_init(&(pool->lock), NULL);
    pthread_cond_init(&(pool->notify), NULL);

    for(int i = 0; i < num_thread; i++){
        pthread_create(&(pool->threads[i]), NULL, mapperExecute, (void*)pool);
        pthread_mutex_lock(&(pool->lock));
        pool->thread_cnt++;
        pool->free_thread_cnt++;

        pool->started++; //
        pthread_mutex_unlock(&(pool->lock));
    }

    // Get tasks
    bool isFinish = false;
    while(!isFinish){

        while (!(pool->free_thread_cnt));
        int dataChunk;
        MPI_Status status;
        MPI_Send( &rank, 1, MPI_INT , 0, REQUEST, MPI_COMM_WORLD);
        MPI_Recv( &dataChunk, 1, MPI_INT, 0, DISPATCH, MPI_COMM_WORLD, &status);

        if(dataChunk == -1){
            isFinish = true;
        }else{
            pthread_mutex_lock(&(pool->lock));
            (*pool->task).push(dataChunk);
            pthread_mutex_unlock(&pool->lock);
        }
    }
}


// Reducer thread
void thread_init(int rank, int num_chunk, int chunk_size, int num_reducer, std::string job_name, std::string output_dir){
    // pthread_t *thread = (pthread_t *)malloc(sizeof(pthread_t));

    bool isFinish = false;
    while(!isFinish){
        int reducerID;
        MPI_Status status;
        MPI_Send( &rank, 1, MPI_INT , 0, REQUEST_REDUCE, MPI_COMM_WORLD);
        MPI_Recv( &reducerID, 1, MPI_INT, 0, DISPATCH_REDUCE, MPI_COMM_WORLD, &status);

        std::string reducerID_s = std::to_string(reducerID);
        if(reducerID == -1){
            isFinish = true;
        }else{
            std::ifstream inter_file("./intermediateFile/reducer" + reducerID_s + ".out");
            std::string line;
            std::vector<Item> word_count;

            while(getline(inter_file, line)){
                if(line == "\n") continue;  //
                size_t pos = line.find(" ");
                std::string word;
                int count;
                word = line.substr(0, pos);
                count = stoi(line.substr(pos+1));
                Item tmp = std::make_pair(word, count);
                word_count.push_back(tmp);
            }
            inter_file.close();

            // Sort function
            // CHANGE SORTING METHOD HERE 
            std::sort(word_count.begin(), word_count.end(), [](const Item &item1, const Item &item2) -> bool
              { return item1.first < item2.first; });


            // Group function
            std::map<std::string, std::vector<Item>>group_pair;
            for(int i = 0; i < word_count.size(); i++){
                // CHANGE GROUPING STRATEGY HERE 
                group_pair[word_count[i].first].push_back(word_count[i]);
            }

            // Reduce function
            std::map<std::string, int>result;
            for(auto group : group_pair) {
                int sum = 0;
                std::vector<Item> words = group.second;
                for(auto it: words) {
                    sum += it.second;
                }
                result[group.first] = sum;
            }

            // Output function
            std::ofstream output_file(output_dir + "/" + job_name + "-" + reducerID_s + ".out");
            for(auto r : result) {
                output_file << r.first << " " << r.second << std::endl;
            }
            output_file.close();
        }
    }
}

int main(int argc, char **argv)
{
    std::chrono::steady_clock::time_point t1, t2;
    t1 = std::chrono::steady_clock::now();
    std::string job_name = std::string(argv[1]);
    int num_reducer = std::stoi(argv[2]);
    int delay = std::stoi(argv[3]);
    std::string input_filename = std::string(argv[4]);
    int chunk_size = std::stoi(argv[5]);
    std::string locality_config_filename = std::string(argv[6]);
    std::string output_dir = std::string(argv[7]);
    std::ofstream log_file(output_dir + "/" + job_name + "-log.out");

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int num_chunk = 0;

    // jobtracker
    if(!rank){
        cpu_set_t cpuset;
        sched_getaffinity(0, sizeof(cpuset), &cpuset);
        int ncpus = CPU_COUNT(&cpuset);

        log_file << std::time(nullptr) << ", Start_Job, " << job_name << ", " << size << ", " << ncpus << ", " <<  num_reducer << ", " << delay << ", " << input_filename << ", " << chunk_size << ", " << locality_config_filename << ", " << output_dir << std::endl;
        // Read the data locality file, generate mapper task queue (taskID, nodeID)
        std::ifstream config(locality_config_filename);
        std::string line;
        std::list<TaskInfo> mapTaskQueue;
        std::list<int> reduceTaskQueue;
        
        int noMoreTask = -1;
        // Generate mapper tasks
        while (getline(config, line)){
            size_t pos = line.find(" ");
            int nodeID, chunkID;
            chunkID = stoi(line.substr(0, pos));
            nodeID = stoi(line.substr(pos+1));

            if(nodeID > size-1) nodeID = (nodeID % (size-1)) + 1;
            TaskInfo tmp = std::make_pair(chunkID, nodeID);
            mapTaskQueue.push_back(tmp);
            num_chunk++;
        }
        config.close();

        // Get mapper requirement from nodes(workers)
        int req_worker;
        MPI_Status status;
        bool isDispatch;
        while(!mapTaskQueue.empty()){

            isDispatch = false;
            MPI_Recv(&req_worker, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, MPI_COMM_WORLD, &status);

            for (std::list<TaskInfo>::iterator it=mapTaskQueue.begin(); it!=mapTaskQueue.end(); it++) {

                if((*it).second == req_worker){

                    log_file << std::time(nullptr) << ", Dispatch_MapTask, " << (*it).first << ", " << req_worker << std::endl;
                    MPI_Send( &((*it).first), 1, MPI_INT, req_worker, DISPATCH, MPI_COMM_WORLD);
                    it = mapTaskQueue.erase(it);
                    it--;
                    isDispatch = true;
                    break;
                }
            }
            if(!isDispatch){
                log_file << std::time(nullptr) << ", Dispatch_MapTask, " << mapTaskQueue.front().first << ", " << req_worker << std::endl;
                MPI_Send( &(mapTaskQueue.front().first), 1, MPI_INT, req_worker, DISPATCH, MPI_COMM_WORLD);
                mapTaskQueue.pop_front();
                isDispatch = true;
            }
        }


        // No more mapper task  // cont'd
        for(int i = 1; i < size; i++){
            // log_file << std::time(nullptr) << ", Complete_MapTask, " << mapTaskQueue.front().first << ", " << req_worker << std::endl;
            MPI_Recv( &req_worker, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, MPI_COMM_WORLD, &status);
            MPI_Send( &noMoreTask, 1, MPI_INT, req_worker, DISPATCH, MPI_COMM_WORLD);
        }


        // Generate reducer tasks
        for(int i = 1; i <= num_reducer; i++){
            reduceTaskQueue.push_back(i);
        }

        // Partition function
        std::chrono::steady_clock::time_point start, end;
        start = std::chrono::steady_clock::now();
        log_file << std::time(nullptr) << ", Start_Shuffle, " << std::endl;
        std::ofstream* intermediateFiles = new std::ofstream[num_reducer];
        for(int i=0; i<num_reducer; i++) {
            std::string path = "./intermediateFile/reducer" + std::to_string(i+1) + ".out";
            intermediateFiles[i] = std::ofstream(path);
        }

        // std::vector<std::vector<Item>> partition_list(num_reducer);
        for(int i = 0; i < num_chunk; i++){
            std::vector<Item> inter_data;
            std::string num_chunk_s = std::to_string(i+1);
            std::ifstream chunk_file("./wordInChunk/chunk" + num_chunk_s + ".out");
            std::string line;
            while(getline(chunk_file, line)){
                if(line == "\n" || line == "") continue;  //
                size_t pos = line.find(" ");
                std::string word;
                int count;
                word = line.substr(0, pos);
                
                count = stoi(line.substr(pos+1));

                Item tmp = std::make_pair(word, count);
                inter_data.push_back(tmp);
            }

            for(auto pair: inter_data) {
                int idx = (int)pair.first[0] % num_reducer;
                intermediateFiles[idx] << pair.first << " " << pair.second << std::endl;
            }
            chunk_file.close();
        }

        for(int i=0; i<num_reducer; i++) {
            intermediateFiles[i].close();
        }
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        log_file << std::time(nullptr) << ", Finish_Shuffle, " << ((double)time_span.count())/1000 << std::endl;


        // Get reducer requirement from nodes(workers)
        while(!reduceTaskQueue.empty()){
            log_file << std::time(nullptr) << ", Dispatch_ReduceTask, " << reduceTaskQueue.front() << ", " << req_worker << std::endl;
            MPI_Recv(&req_worker, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_REDUCE, MPI_COMM_WORLD, &status);
            MPI_Send(&(reduceTaskQueue.front()), 1, MPI_INT, req_worker, DISPATCH_REDUCE, MPI_COMM_WORLD);
            reduceTaskQueue.pop_front();
        }

        // No more reducer task
        for(int i = 1; i < size; i++){
            MPI_Recv( &req_worker, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_REDUCE, MPI_COMM_WORLD, &status);
            MPI_Send( &noMoreTask, 1, MPI_INT, req_worker, DISPATCH_REDUCE, MPI_COMM_WORLD);
        }
        
    }
    // workers
    else{
        cpu_set_t cpuset;
        sched_getaffinity(0, sizeof(cpuset), &cpuset);
        int ncpus = CPU_COUNT(&cpuset);
        // Mapper thread pool init
        threadpool_init(rank, ncpus-1, input_filename, chunk_size, num_reducer);
        // Reducer thread init
        thread_init(rank, num_chunk, chunk_size, num_reducer, job_name, output_dir);
    }
    MPI_Finalize();
    t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    log_file << std::time(nullptr) << ", Finish_Job, " << (double)time_span.count() << std::endl;
    return 0;
}
