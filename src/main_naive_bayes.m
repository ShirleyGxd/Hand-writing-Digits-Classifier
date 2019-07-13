% main_naive_bayes
clear;
addpath('../data/');

% read in the train data and the test data
tr_imgs = loadMNISTImages('train-images-idx3-ubyte');
tr_labs = loadMNISTLabels('train-labels-idx1-ubyte');

te_imgs = loadMNISTImages('t10k-images-idx3-ubyte');
te_labs = loadMNISTLabels('t10k-labels-idx1-ubyte');

% calculate some parameters
[img_size,~]=size(tr_imgs);
[te_num,~]=size(te_labs);
[tr_num,~]=size(tr_labs);

% calculate the feature-binary image
tr_fea=im2bw(tr_imgs ,0);
te_fea=im2bw(te_imgs ,0);


%% train a model

% calculate the prior probability
prior=zeros(1,10);
for i=1:tr_num
    class=tr_labs(i,1);
    prior(class+1)=prior(class+1)+1;
end

% calculate the likelihood/conditional probability of that pixel i of digit j is 1.
lp=zeros(img_size,10);
for i=1:img_size        %the total pixels of a image 
    for j=1:tr_num      %the total train images 
        if(tr_fea(i,j))
            class=tr_labs(j,1);
            lp(i,class+1)=lp(i,class+1)+1;
        end
    end
end

% Laplace Smoothing 
k=1;
for c=1:10
    lp(:,c)=(lp(:,c)+k)*1.0./(prior(c)+k*2);
end

%% classify testing data
final_class=zeros(te_num,1);
for t=1:te_num
    p=zeros(1,10);
    for c=1:10
        p(c)=prior(c);
        for r=1:img_size
            if(te_fea(r,t))    % pixel r of image t is 1.
                p(c)=p(c)*lp(r,c);
            else               % pixel r of image t is 0.
                p(c)=p(c)*(1-lp(r,c));
            end
        end
    end
    [max_value,final_class(t)]=max(p);
    final_class(t)=final_class(t)-1;
end


%% you need to dispaly the accuracy on test data
err=abs(final_class-te_labs);
err_num=sum(im2bw(err,0));         % the times of error classify
accuracy=1-(err_num/te_num);            

disp('The classify Accuracy is:');
disp(accuracy);
