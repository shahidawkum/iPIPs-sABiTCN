clc
clear all

n_protein = 2872;

fileFolder=fullfile('F:\iPIPs-sABiTCN\PIP-Train-PSSM');
dirOutput=dir(fullfile(fileFolder,'*.txt'));
PSSM_XXXX={dirOutput.name}';
PSSM_XXXX = natsortfiles(PSSM_XXXX);
fileNames_PSSM = [];
for i=1:n
    i
	path_way = [fileFolder '\' cell2mat(PSSM_XXXX(i))];
	lujing=cellstr(path_way);
	fileNames_PSSM = [fileNames_PSSM;lujing];
end


%%%%%%%%%%% Features extraction from PSSM %%%%%%%%%%%%%%%% 

for i=1:n_protein
	files_name = cell2mat(fileNames_PSSM(i));

	PSSM_Matrix = Read_Text_files_PSSM(files_name);
    
    %%%%%%%%%%% LBP-PSSM %%%%%%%%%%%%%%%%
    PSSM_IMG = uint8(255 * mat2gray(PSSM_Matrix));
%     imshow(PSSM_IMG);
    LPQfeat=lpq(PSSM_IMG,3);
    LPQ_PSSM_Features_training(i,:)=LPQfeat;
    
    

%%%%%%%%%%%%%%%%%%%%%%%% SAVE FILES %%%%%%%%%%%%%%%%%%%%%%%%%

save LPQ_PSSM_Features_training LPQ_PSSM_Features_training;


%%%% To Create CSV sheet for the data %%%%%%%%%
   
     csvwrite('LPQ_PSSM_Features_training.csv',LPQ_PSSM_Features_training);
      
      
  
