clear

target = '1ft2ep-8';

% load (['/local-scratch/xla193/cluster_video_/output/UCF-101/output-UCF-101-10-',target,'-9-w.mat'])
% load /cs/vml2/xla193/cluster_video/output/UCF-101/mappedX.mat
% load (['/local-scratch/x  la193/cluster_video_/output/UCF-101/cop-kmeans-result-fix-',target,'.mat'])
load (['/local-scratch/xla193/cluster_video_/output/UCF-101/cop-kmeans-fix-veri-8ft.mat'])
load /local-scratch/xla193/cluster_video_/output/UCF-101/UCF-101-gtlabel-10.mat % label
load /local-scratch/xla193/cluster_video_/output/UCF-101/output-UCF-101-10-8ft2ep-all-8-copy1.mat
% load ('/cs/vml2/xla193/cluster_video/output/UCF-101/UCF-101-label-20-0ft.mat', 'kmcenters')
addpath(genpath('/cs/vml2/xla193/dmmc/code/visualization'))
s = ['fix initialization - ',target];
% margin = 1000;
% centers = zeros(10,4096);
% centers(1,:) = data(1,:);
% centers(2,:) = data(146,:);
% centers(3,:) = data(260,:);
% centers(4,:) = data(405,:);
% centers(5,:) = data(537,:);
% centers(6,:) = data(645,:);
% centers(7,:) = data(800,:);
% centers(8,:) = data(950,:);
% centers(9,:) = data(1084,:);
% centers(10,:) = data(1215,:);


[idx, kmcenters, sumd, D] = kmeans(data, 10, ...
     'Replicates',100,'Display','final','MaxIter',300);

% [idx, kmcenters, sumd, D] = kmeans(data, 10, ...
%        'Display','final','MaxIter',300,'Start',centers);
% pdlabels = double(pdlabels');
pdlabels = idx;
gtlabels = label;
flag = 0;
beta = 1;

%%%%%%%%%%%%%%%%%%%%visualize%%%%%%%%%%%%%%%%%%%
% initial_dims = 100;
% no_dims = 2;
% perplexity = 30;
% 
% mappedX = tsne(data, [], no_dims, initial_dims, perplexity);
% figure
% gscatter(mappedX(:,1), mappedX(:,2), pdlabels);
% title(s);
% figure
% gscatter(mappedX(:,1), mappedX(:,2), gtlabels);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

unigt = unique(gtlabels);
count_gt = histc(gtlabels, unigt);
nugt = length(unigt);
ngts = length(gtlabels);

unipd = unique(pdlabels);
unipd(unipd == 0) = [];
count_pd_nmi = histc(pdlabels, unipd);
nupd = length(unipd);
npds = length(pdlabels);

% intersection
count_int = zeros(nugt, nupd);
for i = 1:nugt
    for j = 1:nupd
        count_int(i, j) = sum((gtlabels == unigt(i)) & (pdlabels == unipd(j)));
    end
end

logterm = repmat(count_gt, [1, nupd]) .* repmat(count_pd_nmi', [nugt, 1]);
logterm = ngts .* count_int ./ logterm;
MIs = count_int .* log2(logterm + eps) ./ npds;
MI = sum(MIs(:));

% normalization
prob_gt = count_gt ./ ngts;
Hgt = - sum(prob_gt .* log2(prob_gt));

prob_pd = count_pd_nmi ./ npds;
Hpd = - sum(prob_pd .* log2(prob_pd));

nmi = 2 * MI / (Hgt + Hpd);

count_pd = sum(count_int, 1);

sort_count_int = sort(count_int, 1, 'descend');
cum_count = cumsum(sort_count_int, 1);

pa_topk = sum(cum_count, 2) / sum(count_pd);
pp_topk = mean(cum_count ./ repmat(count_pd, [nugt, 1]), 2);


[nmaxgt, maxgt] = max(count_int, [], 1);
pa = sum(nmaxgt) / sum(count_pd);
pp = mean(nmaxgt ./ count_pd);

TPFP = count_pd_nmi .* (count_pd_nmi - 1) / 2;
TPFP = sum(TPFP);

TP = count_int .* (count_int - 1) / 2;
TP = sum(TP(:));

FP = TPFP - TP;

sumcount_int = cumsum(count_int, 2);
FN = sumcount_int(:, 1:end - 1) .* count_int(:, 2:end);
FN = sum(FN(:));
P = TP / (TP + FP);
R = TP / (TP + FN);

fm = ((beta * beta + 1) * P * R) / (beta * beta * P + R);
sumcount_int = cumsum(count_int, 2);
FN = sumcount_int(:, 1:end - 1) .* count_int(:, 2:end);
FN = sum(FN(:));
TPFPTNFN = sum(count_pd_nmi) * (sum(count_pd_nmi) - 1) / 2;
ri = 1 - ((FP + FN) / TPFPTNFN);

meas.ri = ri
meas.pa = pa
meas.nmi= nmi

% str = {['ri:   ', num2str(ri)], ['pa:  ', num2str(pa)], ['nmi:', num2str(nmi)], ['margin:', num2str(margin)]};
% dim = [.15 .6 .3 .3];
% annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on');
% save /local-scratch/xla193/cluster_video_/output/UCF-101/outputlabelsingle-UCF-101-10-0ftuser.mat gtlabels pdlabels mappedX meas

