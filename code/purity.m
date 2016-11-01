clear
load /cs/vml2/xla193/cluster_video/output/UCF-101/outputsingle-UCF-101-10-uf1.mat
load /cs/vml2/xla193/cluster_video/output/UCF-101/cop-kmeans-result1.mat
load /cs/vml2/xla193/cluster_video/output/UCF-101/UCF-101-gtlabel-10.mat % label
% load ('/cs/vml2/xla193/cluster_video/output/UCF-101/UCF-101-label-20-0ft.mat', 'kmcenters')
addpath(genpath('/cs/vml2/xla193/dmmc/code/visualization'))

% [idx, kmcenters, sumd, D] = kmeans(data, 10, ...
%      'Replicates',100,'Display','final','MaxIter',300);

% [idx, kmcenters, sumd, D] = kmeans(data, 20, ...
%        'Display','final','MaxIter',300,'Start',centers);
pdlabels = double(pdlabels');
gtlabels = label;
flag = 0;
beta = 1;

%%%%%%%%%%%%%%%%%%%%visualize%%%%%%%%%%%%%%%%%%%
initial_dims = 100;
no_dims = 2;
perplexity = 30;

mappedX = tsne(data, [], no_dims, initial_dims, perplexity);
figure
gscatter(mappedX(:,1), mappedX(:,2), pdlabels);
figure
gscatter(mappedX(:,1), mappedX(:,2), gtlabels);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if flag
%     ids = find(chosedPoints(:,2) >= 0);
%     gtlabels = labelgt(ids,:);
%     pdlabels = labels(ids,:);
% else
%     gtlabels = label;
%     pdlabels = data;
% end

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

save /cs/vml2/xla193/cluster_video/output/UCF-101/outputlabelsingle-UCF-101-10-0ftuser.mat gtlabels pdlabels mappedX meas

