file1 = '../data/movieqa/copy_full_qa_train.txt'
file2 = '../data/movieqa/clean_qa_full_train.txt'
with open(file1,'r',encoding='utf-8') as f1:
    with open(file2,'r',encoding='utf-8') as f2:
        cnt=1
        diff=0
        for l1,l2 in zip(f1,f2):
            if(l1!=l2):
                diff+=1
                print(cnt)
                print(l1)
                print(l2)
                #break
            cnt+=1
        print(diff)