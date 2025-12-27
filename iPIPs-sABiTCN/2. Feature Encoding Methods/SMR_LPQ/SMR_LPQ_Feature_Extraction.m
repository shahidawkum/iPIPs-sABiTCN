clc
clear all
load('energy_20.mat');

feature_train=[];


[data, sequence] = fastaread('PIPs_Training.txt');

Total_Seq_train=size(sequence,2);
for i=1:(Total_Seq_train)
     i
    SEQ=sequence(i);
    SEQ=cell2mat(SEQ);
    P=SMR(SEQ,energy_20);
	P=P';
    P = uint8(255 * mat2gray(P));
	FF=lpq(P,3);
    features(i,:)=FF;
end

%%%%%%%%%%%%%%%%%%%%%%%% SAVE FILES %%%%%%%%%%%%%%%%%%%%%%%%%

SMR_LPQ_Features_Training=[features];
save SMR_LPQ_Features_Training SMR_LPQ_Features_Training;

%%%% To Create CSV sheet for the data %%%%%%%%%
   
      csvwrite('SMR_LPQ_Features_Training.csv',SMR_LPQ_Features_Training);
     


