
clear;
input_dir = '../cvpr21-absorption/data2-glass/';
filename = dir([input_dir '3/t/']);
ks = 50;
f  = ones(ks+1,ks+1)/(ks+1)/(ks+1);
k=0;
tx = 500;
th = 10;
th2 = 0.0;
tt = 0.95;
for i = 1:20
    img3 = double(rgb2gray(imread([input_dir '3/syn/' filename(i+2).name])));
    img5 = double(rgb2gray(imread([input_dir '5/syn/' filename(i+2).name])));
    img10 = double(rgb2gray(imread([input_dir '10/syn/' filename(i+2).name])));
    img3t = double(rgb2gray(imread([input_dir '3/t/' filename(i+2).name])));
    img5t = double(rgb2gray(imread([input_dir '5/t/' filename(i+2).name])));
    img10t = double(rgb2gray(imread([input_dir '10/t/' filename(i+2).name])));
    
    img3r = double(rgb2gray(imread([input_dir '3/r/' filename(i+2).name])));
    img5r = double(rgb2gray(imread([input_dir '5/r/' filename(i+2).name])));
    img10r = double(rgb2gray(imread([input_dir '10/r/' filename(i+2).name])));
    
    [h,w,c] = size(img3);
    
    
    mssim = ssim_modify(uint8(img3t), uint8(img5t));
    e35 = mssim;
    mssim = ssim_modify(uint8(img3t), uint8(img10t));
    e310 = mssim;
    mssim = ssim_modify(uint8(img5t), uint8(img10t));
    e510 = mssim;
    
    
    c3 = conv2(img3r, f, 'valid');
    c5 = conv2(img3r, f,'valid');
    c10 = conv2(img10r, f, 'valid');
    
    if e35>tt
        cc = c3+c5;
        
        index_min = find(cc==min(cc(:)));
        index_i = ceil(index_min(1)/(h-ks));
        index_j = index_min(1) - (index_i-1)*(h-ks);
        a = img3(index_j:index_j+ks,index_i:index_i+ks,1);
        b = img5(index_j:index_j+ks,index_i:index_i+ks,1);
        at = img3t(index_j:index_j+ks,index_i:index_i+ks,1);
        bt = img3t(index_j:index_j+ks,index_i:index_i+ks,1);
        ar = img3r(index_j:index_j+ks,index_i:index_i+ks,1);
        br = img5r(index_j:index_j+ks,index_i:index_i+ks,1);
        
        ra = (double(a)-mean(double(a(:))))./(double(at)-mean(double(at(:))));
        rb = (double(b)-mean(double(b(:))))./(double(bt)-mean(double(bt(:))));
        
        
        
        ma = double(ra<1&ra>th2&ar<th);
        mb = double(rb<1&rb>th2&br<th);
        if sum(ma(:))>tx && sum(mb(:))>tx
            k = k+1;
            label(k,:) = [i,3,5];
            results(k,:) = [sum(sum(ra.*ma))./sum(ma(:)),sum(sum(rb.*mb))./sum(mb(:))];
        end
    end
    if e310>tt
        cc = c3+c10;
        
        index_min = find(cc==min(cc(:)));
        index_i = ceil(index_min(1)/(h-ks));
        index_j = index_min(1) - (index_i-1)*(h-ks);
        a = img3(index_j:index_j+ks,index_i:index_i+ks,1);
        b = img10(index_j:index_j+ks,index_i:index_i+ks,1);
        at = img3t(index_j:index_j+ks,index_i:index_i+ks,1);
        bt = img3t(index_j:index_j+ks,index_i:index_i+ks,1);
        ar = img3r(index_j:index_j+ks,index_i:index_i+ks,1);
        br = img10r(index_j:index_j+ks,index_i:index_i+ks,1);
        
        ra = (double(a)-mean(double(a(:))))./(double(at)-mean(double(at(:))));
        rb = (double(b)-mean(double(b(:))))./(double(bt)-mean(double(bt(:))));
        
        
        
        ma = double(ra<1&ra>th2&ar<th);
        mb = double(rb<1&rb>th2&br<th);
        
        if sum(ma(:))>tx && sum(mb(:))>tx
            k = k+1;
            label(k,:) = [i,3,10];
            results(k,:) = [sum(sum(ra.*ma))./sum(ma(:)),sum(sum(rb.*mb))./sum(mb(:))];
            
        end
    end
    if e510>tt
        cc = c5+c10;
        
        index_min = find(cc==min(cc(:)));
        index_i = ceil(index_min(1)/(h-ks));
        index_j = index_min(1) - (index_i-1)*(h-ks);
        a = img5(index_j:index_j+ks,index_i:index_i+ks,1);
        b = img10(index_j:index_j+ks,index_i:index_i+ks,1);
        at = img5t(index_j:index_j+ks,index_i:index_i+ks,1);
        bt = img5t(index_j:index_j+ks,index_i:index_i+ks,1);
        ar = img5r(index_j:index_j+ks,index_i:index_i+ks,1);
        br = img10r(index_j:index_j+ks,index_i:index_i+ks,1);
        
        ra = (double(a)-mean(double(a(:))))./(double(at)-mean(double(at(:))));
        rb = (double(b)-mean(double(b(:))))./(double(bt)-mean(double(bt(:))));
        
        
        
        ma = double(ra<1&ra>th2&ar<th);
        mb = double(rb<1&rb>th2&br<th);
        
        if sum(ma(:))>tx && sum(mb(:))>tx
            
            k = k+1;
            label(k,:) = [i,5,10];
            results(k,:) = [sum(sum(ra.*ma))./sum(ma(:)),sum(sum(rb.*mb))./sum(mb(:))];
        end
    end
    
end
figure;plot(results);
