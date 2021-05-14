function [map_coe_T, map_coe_R] = Get_Map(h,w)

FoV_list = [19.45, 23.12,30.08,33.40,39.60,48.46,65.47,73.74];
thick_list = [3,6,9]*0.001;

while 1
    g_dis = 0;
    while g_dis<0.2
        g_dis = rand(1)*(5-0.2)+0.2;
    end
    g_size = 0;
    while g_size<0.4
        g_size = rand(1)*(3-0.4)+0.4;
    end
    g_angle1 = 90;
    while abs(g_angle1)>60
        g_angle1 =  10*(randn(1,1));
    end
    g_angle2 = 90;
    while abs(g_angle2)>15
        g_angle2 =  4*(randn(1,1));
    end
    FoV = rand(1)*(73.74-19.45)+19.45;
    
    
    thickness = rand(1)*0.007+0.003;
    
    cor_x2 = tan((-FoV/2 + abs(g_angle1))/180*pi)*g_dis;
    cor_x3 = tan((FoV/2 + abs(g_angle1))/180*pi)*g_dis;
    
    if cor_x3-cor_x2 < g_size
        break;
    end
end


n2 = 1.474;

map_T = ComputeThetaMap(w, h, g_dis, g_angle1, g_angle2, FoV,g_size);
[map_A, map_B] = ComputeMap(map_T, n2);

cos_map_T2 = 1./sqrt(1-(1/n2*sin(map_T)).^2);
k_c = rand(1)*(32-4)+4;
map_alpha = exp(-k_c*thickness*cos_map_T2);


map_coe_R = map_B + map_B.*(map_A.*map_A.*map_alpha.*map_alpha)./(1-map_B.*map_B.*map_alpha.*map_alpha);
map_coe_T = (map_A.*map_A.*map_alpha)./(1-map_B.*map_B.*map_alpha.*map_alpha);
end

function map_T = ComputeThetaMap(w, h, g_dis, angle1, angle2, FoV,g_size)
% cor_x1 = g_dis*(tan(abs(g_angle)/180*pi));

cor_x2 = tan((-FoV/2 + abs(angle1))/180*pi)*g_dis;
cor_x3 = tan((FoV/2 + abs(angle1))/180*pi)*g_dis;
centerX = 0;
centerY = 0;
centerZ = w/2/tan(FoV/2/180*pi);
if cor_x3-cor_x2 > g_size
    step = g_size*cos(angle1/180*pi)/w;
else
    step = 1;
end
map_T = zeros(h,w);
n = [0,0,0];
nv = [sin(angle1/180*pi)*cos(angle2/180*pi),sin(angle1/180*pi)*sin(angle2/180*pi),cos(angle1/180*pi)];
for i = -w/2 : w/2-1
    for j=-h/2 : h/2-1
        n(1) = (i+0.5)*step+centerX;
        n(2) = (j+0.5)*step+centerY;
        n(3) = centerZ;
        n = n./norm(n);
        map_T(j+h/2+1, i+w/2+1) = acos(nv*n');
    end
end
end


function [map_A, map_B] = ComputeMap(map_T, n2)
n1 = 1;
Rs1 = @(x)((n1*cos(x)-n2*sqrt(1-(n1/n2*sin(x)).^2))/(n1*cos(x)+n2*sqrt(1-(n1/n2*sin(x)).^2))).^2;
% Ts1 = @(x)1-Rs1(x);
Rp1 = @(x)((n2*cos(x)-n1*sqrt(1-(n1/n2*sin(x)).^2))/(n2*cos(x)+n1*sqrt(1-(n1/n2*sin(x)).^2))).^2;
R = @(x) 0.5*(Rs1(x) + Rp1(x));
%
% n3 = n2;
% n4 = n1;
% Rs2 = @(x)((n3*cos(x)- n4*sqrt(1-(n3/n4*sin(x)).^2))/(n3*cos(x)+n4*sqrt(1-(n3/n4*sin(x)).^2))).^2;
% Ts2 = @(x)1-Rs2(x);
% Rp2 = @(x)((n4*cos(x)- n3*sqrt(1-(n3/n4*sin(x)).^2))/(n4*cos(x)+n3*sqrt(1-(n3/n4*sin(x)).^2))).^2;
% Tp2 = @(x)1-Rp2(x);
%
% Tp1Tp2 = @(x) Tp1(x)*Tp2(asin(n1/n2*sin(x)));
% Ts1Ts2 = @(x) Ts1(x)*Ts2(asin(n1/n2*sin(x)));
%
% T1T2 = @(x) (Tp1Tp2(x) + Ts1Ts2(x))/2;
% R1 = @(x) 1 - T1T2(x);%(Rs1(x) + Rp1(x))/2;

map_A = zeros(size(map_T));
map_B = zeros(size(map_T));
[h, w, c] = size(map_T);
for i=1:h
    for j=1:w
        map_A(i,j) = 1-R(map_T(i,j));
        map_B(i,j) = R(map_T(i,j));
    end
end
end


