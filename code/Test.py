file1 = '../data/movieqa/copy_entities.txt'
file2 = '../data/movieqa/clean_entities.txt'
with open(file1,'r') as f1:
    with open(file2,'r') as f2:
        cnt=1
        diff=0
        for l1,l2 in zip(f1,f2):
            if(l1!=l2):
                diff+=1
                print(cnt,l1,l2)
                break
            cnt+=1
        print(diff)