
file_directory = "../test/"
num_reducer = int(input('num_reducer: '))
testcase = input('testcase: ')

words = dict()

# for i in range(1, num_reducer+1, 1):
#     for j in range(1, chunk+1, 1):
#         file = str(j) + '_' + str(i) + '.txt'
#         with open(file_directory+file, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 tmp = line.split(' ')
#                 word = tmp[0]
#                 num = int(tmp[1])
#                 if word not in words:
#                     words[word] = num
#                 else:
#                     words[word] += num

for i in range(1, num_reducer+1, 1):
    file = "TEST" + testcase + "-" + str(i) + ".out"
    with open(file_directory+file, 'r') as f:
        lines = f.readlines();
        for line in lines:
            tmp = line.split(' ')
            word = tmp[0]
            num = int(tmp[1])
            if word not in words:
                words[word] = num
            else:
                words[word] += num


# print(sorted(words.items(), key = lambda kv:(kv[1], kv[0]))) 

ans = dict()
file_directory = "../testcases/" + testcase + "_sample_ans/"

for i in range(1, num_reducer+1, 1):
    file = "TEST" + testcase + "-" + str(i) + ".out"
    with open(file_directory+file, 'r') as f:
        lines = f.readlines();
        for line in lines:
            tmp = line.split(' ')
            word = tmp[0]
            num = int(tmp[1])
            if word not in ans:
                ans[word] = num
            else:
                ans[word] += num


# print(sorted(ans.items(), key = lambda kv:(kv[1], kv[0]))) 

items = ans.items()
tmp = words.items()
# print(tmp)
for i in items:
    if i not in tmp:
        print(i)
# print(len(items))
# print(len(tmp))