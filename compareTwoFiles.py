
firstfile = 'swedish_test.tsv'
otherfile = '/lhome/larsbun/git-projects/multiged-2023/swedish/sv_swell_test_unlabelled.tsv'

fo = open(firstfile, 'r')
fs = open(otherfile, 'r')

forl = fo.readlines()
fsrl = fs.readlines()

counter = 0

for i in range(len(forl)):
    fw_f = forl[i].split('\t')[0].strip()
    fw_s = fsrl[i].split('\t')[0].strip()

    if not fw_f == fw_s:
        print ("Not matching", i, fw_s, fw_f, forl[i-1], forl[i], forl[i+1])
        counter += 1
        if counter > 5:
            break


fo.close()
fs.close()