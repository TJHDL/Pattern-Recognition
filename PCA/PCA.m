name1 = 'E:\\学校事务\\学校课程\\模式识别导论\\作业\\1751700-黄定梁-第七次作业\\YaleFaceDatabase\\S%d.jpg';  %原始数据路径
name2 = 'E:\\学校事务\\学校课程\\模式识别导论\\作业\\1751700-黄定梁-第七次作业\\result\\S%d.jpg';    %降维后数据路径
[k, n] = deal(20, 165);              %取前20个特征向量，共165张图片
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