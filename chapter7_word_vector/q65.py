
count = 0
correct = 0
ans = {
    'semantic': [],
    'syntactic': [],
}
with open('files/output-q64.txt') as f:
    key = ''
    for line in f.readlines():
        if line[0] == ':':
            key = 'syntactic' if 'gram' in line else 'semantic'
        else:
            words = line.split()
            ans[key].append(words[3] == words[4])

print(f'semantic: {ans["semantic"].count(True) / len(ans["semantic"])}')
print(f'syntactic: {ans["syntactic"].count(True) / len(ans["syntactic"])}')
