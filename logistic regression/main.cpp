#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<vector>
#include"mnist.h"
#include"logistic.h"
using namespace std;

int main()
{
    v2uc image,image2,test,test2;
    vi label,label2,testlabel,testlabel2;

    mnist mst;
    int postive=2,negative=5;

    //‘ÿ»Î—µ¡∑ºØ
    mst.load(image2,label2,mst.train_image,mst.train_label);
    int total=0;
    for(int i=0;i<label2.size();i++)
        total+=label2[i]==postive||label2[i]==negative;
    image.resize(total);
    label.resize(total);
    int cur=0;
    for(int i=0;i<label2.size();i++) //»•µÙ £”‡ ˝◊÷
        if(label2[i]==postive||label2[i]==negative) image[cur]=image2[i],label[cur]=label2[i]==postive,cur++;

    logistic<float> lgsc;
    lgsc.init(image[0].size());
    lgsc.train(image,label,0.001);


    //‘ÿ»Î≤‚ ‘ºØ
    mst.load(test2,testlabel2,mst.test_image,mst.test_label);
    total=0;
    for(int i=0;i<testlabel2.size();i++)
        total+=testlabel2[i]==postive||testlabel2[i]==negative;
    test.resize(total);
    testlabel.resize(total);
    cur=0;
    for(int i=0;i<testlabel2.size();i++) //»•µÙ £”‡ ˝◊÷
        if(testlabel2[i]==postive||testlabel2[i]==negative) test[cur]=test2[i],testlabel[cur]=testlabel2[i]==postive,cur++;

    printf("Correct Rate: %.3f%%\n",lgsc.predict(image,label)*100);
    printf("Correct Rate: %.3f%%\n",lgsc.predict(test,testlabel)*100);
    //lgsc.show("weight69.txt");
	return 0;
}
