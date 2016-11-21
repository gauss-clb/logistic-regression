#ifndef LOGISTIC_H_INCLUDED
#define LOGISTIC_H_INCLUDED
#include<cmath>

template<class T>
class logistic{
public:

    vector<T> weight; //学习参数
    T learningRate; //学习率
    T eps; //迭代终止的精度

    //初始化全0
    void init(int dimension){ //特征向量维数,加偏置
        weight.clear();
        weight.resize(dimension);
        for(int i=0;i<dimension;i++)
            weight[i]=0;
    }

    template<typename T1>
    T sigmoid(vector<T1> &feature){
        T x=0; //x=w0*x0+w1*x1+...+wn*xn
        for(int i=0;i<weight.size();i++)
            x+=weight[i]*feature[i];
        return 1/(1+exp(-x));
    }

    T length(vector<T> &temp){
        T len=0;
        for(int i=0;i<temp.size();i++)
            len+=temp[i]*temp[i];
        return sqrt(len);
    }

    //全局梯度
    template<typename T1>
    bool gradient(vector<vector<T1> > &feature,vi &label){
        assert(feature.size()==label.size());
        vector<T> temp(weight.size(),0);
        for(int i=0;i<weight.size();i++)
            for(int j=0;j<feature.size();j++)
                temp[i]+=feature[j][i]*(label[j]-sigmoid(feature[j]));
        for(int i=0;i<weight.size();i++)
            weight[i]+=learningRate*temp[i];
        return length(temp)<eps;
    }

    //随机梯度
    template<typename T1>
    bool stochastic_gradient(vector<vector<T1> > &feature,vi &label){
        assert(feature.size()==label.size());
        for(int j=0;j<feature.size();j++){
            vector<T> temp(weight.size(),0);
            for(int i=0;i<weight.size();i++)
                temp[i]+=feature[j][i]*(label[j]-sigmoid(feature[j]));
            for(int i=0;i<weight.size();i++)
                weight[i]+=learningRate*temp[i];
            //if(length(temp)<eps) return true;
        }
        return false;
    }

    //批梯度
    template<typename T1>
    bool batch_gradient(vector<vector<T1> > &feature,vi &label,int batchSize=100){
        assert(feature.size()==label.size());
        vector<int> index(feature.size(),0);
        for(int i=1;i<index.size();i++) index[i]=i;
        random_shuffle(index.begin(),index.end());
        vector<T> temp(weight.size(),0);
        for(int j=0,k=-1;j<feature.size();j++){
            if(++k==batchSize){
                if(length(temp)<eps) return true;
                for(int i=0;i<weight.size();i++){
                    weight[i]+=learningRate*temp[i]/batchSize;
                    temp[i]=0;
                }
                k=0;
            }
            for(int i=0;i<weight.size();i++)
                temp[i]+=feature[index[j]][i]*(label[index[j]]-sigmoid(feature[index[j]]));
        }
        return false;
    }


    //训练
    template<typename T1>
    void train(vector<vector<T1> > &feature,vi &label,T _learningRate,int maxIter=1,T _eps=1e-6){
        learningRate=_learningRate;
        eps=_eps;
        for(int i=0;i<maxIter;i++){
            //if(stochastic_gradient(feature,label)) break;
            if(batch_gradient(feature,label)) break;
        }
            //stochastic_gradient(feature,label);
    }

    //评估测试集准确率
    template<typename T1>
    double predict(vector<vector<T1> > &testSet,vi &label){
        int identifier;
        int correct=0;
        for(int i=0;i<testSet.size();i++){
            identifier=sigmoid(testSet[i])>=0.5;
            correct+=identifier==label[i];
            //printf("Test %d: predict: %d, fact: %d\n",i+1,identifier,label[i]);
        }
        return correct*1.0/label.size();
    }

    void show(string path){
        FILE* file;
        if((file=fopen(path.c_str(),"w"))==NULL){
            printf("Can't open the %s.\n",path.c_str());
            return;
        }
        fprintf(file,"28 28\n");
        for(int i=0;i<weight.size();i++)
            fprintf(file,"%.3f ",weight[i]);
        fprintf(file,"\n");
        fclose(file);
    }

};

#endif // LOGISTIC_H_INCLUDED
