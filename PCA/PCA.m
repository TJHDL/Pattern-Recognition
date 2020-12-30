name1 = 'E:\\ѧУ����\\ѧУ�γ�\\ģʽʶ����\\��ҵ\\1751700-�ƶ���-���ߴ���ҵ\\YaleFaceDatabase\\S%d.jpg';  %ԭʼ����·��
name2 = 'E:\\ѧУ����\\ѧУ�γ�\\ģʽʶ����\\��ҵ\\1751700-�ƶ���-���ߴ���ҵ\\result\\S%d.jpg';    %��ά������·��
[k, n] = deal(20, 165);              %ȡǰ20��������������165��ͼƬ
M = read_img(name1, n);
M = P_C_A(M, k, n);
write_img(name2, M, n)
 
function [M] = read_img(name, n)
    M = zeros(243*320, n);
    for i = 1:n 
        file_name = sprintf(name, i);
        image = imread(file_name);    
        M(:,i) = image(:);  
    end
end
function [M] = P_C_A(M, k, n)
    F = pca(M);        
    F(:, k+1:n) = 0;   
    M = M * (F * F');  
end
function [] = write_img(name, M, n)
    for i = 1:n
        file_name = sprintf(name, i);
        img = reshape(M(:,i), 243, 320);
        img = mat2gray(img);                   
        imwrite(img, file_name);
    end
end