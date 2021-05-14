clear;
dir1 = '.\input_t\';
dir2 = '.\rimage\';
outdir1 = '.\syn\';
outdir2 = '.\r\';
outdir3 = '.\t\';
mkdir(outdir1);
mkdir(outdir2);
mkdir(outdir3);

filelist = dir(dir1);
f = fopen('AB.txt', 'w');

count = 1;
for i = 1:size(filelist,1)
    imagefullpath = [dir1 filelist(i).name];
    [a,b,c]=fileparts([pwd, dir1 filelist(i).name]);
    
    if ~strcmp(c, '.jpg')
        continue;
    end
   
   
    img1 = imread([dir1,filelist(i).name]);
    subdir = dir(dir2);
    R = randi([3, length(subdir)-1]);
    img2 = imread(fullfile(dir2, subdir(R).name));
    
    if randi([1,2])==1
        img2 = flip(img2, 1);
    end
    if randi([1,2])==1
        img2 = flip(img2, 2);
    end
    
    [h, w, c] = size(img1); 
    
    img2 = imresize(img2, [h,w]);
    
    [map_A, map_B] = Get_Map(h,w);
    
    A = mean(map_A(:));
    B = unifrnd(0.1, 0.6);
    
    img = double(img1).* map_A + double(img2)*B;
    img(img>255)=255;
    img_mix = uint8(img);
    
    img_mix = imresize(img_mix, [256,256]);
    img1 = imresize(img1, [256,256]);
    img2 = imresize(img2, [256,256]);
    
    imwrite(img_mix, [outdir1 num2str(count,'%05d') '.jpg']);
    imwrite(img2, [outdir2 num2str(count,'%05d') '.jpg']);
    imwrite(img1, [outdir3 num2str(count,'%05d') '.jpg']);
    
    disp(num2str(count));
    fprintf(f, [num2str(count,'%05d') '.jpg' '\t%f\t%f\n'], A, B);
    count = count + 1;
end
fclose(f);

