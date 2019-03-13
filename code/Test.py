file1 = '../data/movieqa/copy_idx.txt'
file2 = '../data/movieqa/full_idx.txt'
with open(file1,'r',encoding='utf-8') as f1:
    with open(file2,'r',encoding='utf-8') as f2:
        cnt=1
        diff=0
        for l1,l2 in zip(f1,f2):
            if(l1.split("\t")[0]!=l2.split("\t")[0]):
                diff+=1
                print(cnt)
                print(l1)
                print(l2)
                if diff>10:
                    break
            cnt+=1
        print(diff)