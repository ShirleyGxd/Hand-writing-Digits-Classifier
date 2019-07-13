% main_svm
clear;
addpath('../data/');
addpath('../libsvm-3.21/libsvm-3.21/matlab');

% read in the train data and the test data
tr_imgs = loadMNISTImages('train-images-idx3-ubyte');
tr_labs = loadMNISTLabels('train-labels-idx1-ubyte');

te_imgs = loadMNISTImages('t10k-images-idx3-ubyte');
te_labs = loadMNISTLabels('t10k-labels-idx1-ubyte');

% calculate some parameters
[img_size,~]=size(tr_imgs);
[te_num,~]=size(te_labs);
[tr_num,~]=size(tr_labs);

%% calculate the feature-lbp, each row is a feature of an image
% tr_fea=zeros(tr_num,256);
% for t=1:tr_num
%     img=reshape(tr_imgs(:,t),28,28);
%     tr_fea(t,:)=lbp(img);
% end
% 
% te_fea=zeros(te_num,256);
% for t=1:te_num
%     img=reshape(te_imgs(:,t),28,28);
%     te_fea(t,:)=lbp(img);
% end
% save('trainFeature','tr_fea');
% save('testFeature','te_fea');

% To save time, directly load the computed data
load('trainFeature');
load('testFeature');

%% use the lib to get the maximum accurancy, get that: accuracy = 83.51%
% model=svmtrain(tr_labs,tr_fea);
% [final_class,accuracy]=svmpredict(te_labs,te_fea,model);



%% classify the train samples
% num=zeros(1,10);
% class_fea=cell(1,10);
% for k=1:tr_num
%     label=tr_labs(k)+1;
%     num(label)=num(label)+1;
%     class_fea{1,label}(num(label),:)=tr_fea(k,:);
% end
% save('classFeature','class_fea');
% save('NumberOfEachClass','num');

% To save time, directly load the computed data
load('classFeature');
load('NumberOfEachClass');

%% train the SVM model by using One VS One
% model=cell(1,45);
% times=0;
% for i=1:9
%     for j=i+1:10
%         times=times+1;
%         y=[ones(num(i),1)*(i-1);ones(num(j),1)*(j-1)];
%         trainFea=[class_fea{1,i};class_fea{1,j}];
%         model{1,times}=svmtrain(y,trainFea);
%     end
% end
% save('model','model');

% To save time, directly load the trained model
load('model');

%% Predict by using the established SVM model
class=zeros(te_num,45);
for k=1:45
    class(:,k)=svmpredict(te_labs,te_fea,model{1,k});
end

final_class=mode(class,2);

% save('predictedLabel','final_class');

% To save time, you can alse directly load the predicted labels
% load('predictedLabel');

%% you need to display the accuracy on test data
accuracy=sum(final_class==te_labs)/te_num;            

disp('The classify Accuracy is:');
disp(accuracy);