function handles= plot_results(model,truth,meas,est)

[X_track,k_birth,k_death]= extract_tracks(truth.X,truth.track_list,truth.total_tracks);

labelcount= countestlabels();
colorarray= makecolorarray(labelcount);
est.total_tracks= labelcount;
est.track_list= cell(truth.K,1);
for k=1:truth.K
    for eidx=1:size(est.X{k},2)
        est.track_list{k} = [est.track_list{k} assigncolor(est.L{k}(:,eidx))];
    end
end
[Y_track,l_birth,l_death]= extract_tracks(est.X,est.track_list,est.total_tracks);

%plot ground truths
limit= [ model.range_c(1,1) model.range_c(1,2) model.range_c(2,1) model.range_c(2,2) ];
figure; truths= gcf; hold on;
for i=1:truth.total_tracks
    Pt= X_track(:,k_birth(i):1:k_death(i),i); Pt=Pt([1 3],:);
    plot( Pt(1,:),Pt(2,:),'k-'); 
    plot( Pt(1,1), Pt(2,1), 'ko','MarkerSize',6);
    plot( Pt(1,(k_death(i)-k_birth(i)+1)), Pt(2,(k_death(i)-k_birth(i)+1)), 'k^','MarkerSize',6);
end
axis equal; axis(limit); title('Ground Truths');

%plot tracks and measurments in x/y
figure; clf; tracking= gcf; hold on;

%plot x measurement
subplot(211); box on; 

for k=1:meas.K
    if ~isempty(meas.Z{k})
        hlined= line(k*ones(size(meas.Z{k},2),1),meas.Z{k}(1,:),'LineStyle','none','Marker','x','Markersize',5,'Color',0.7*ones(1,3));
    end   
end

%plot x track
for i=1:truth.total_tracks
    Px= X_track(:,k_birth(i):1:k_death(i),i); Px=Px([1 3],:);
    hline1= line(k_birth(i):1:k_death(i),Px(1,:),'LineStyle','-','Marker','none','LineWidth',1,'Color',0*ones(1,3));
end

%plot x estimate
for t=1:size(Y_track,3)
    hline2= line(1:truth.K,Y_track(1,:,t),'LineStyle','none','Marker','.','Markersize',8,'Color',colorarray.rgb(t,:));
end

%plot y measurement
subplot(212); box on;
    
for k=1:meas.K
    if ~isempty(meas.Z{k})
        yhlined= line(k*ones(size(meas.Z{k},2),1),meas.Z{k}(2,:),'LineStyle','none','Marker','x','Markersize',5,'Color',0.7*ones(1,3));
    end
end

%plot y track
for i=1:truth.total_tracks
        Py= X_track(:,k_birth(i):1:k_death(i),i); Py=Py([1 3],:);
        yhline1= line(k_birth(i):1:k_death(i),Py(2,:),'LineStyle','-','Marker','none','LineWidth',1,'Color',0*ones(1,3));
end

%plot y estimate
for t=1:size(Y_track,3)
    hline2= line(1:truth.K,Y_track(3,:,t),'LineStyle','none','Marker','.','Markersize',8,'Color',colorarray.rgb(t,:));
end

subplot(211); xlabel('Time'); ylabel('x-coordinate (m)');
set(gca, 'XLim',[1 truth.K]); set(gca, 'YLim',model.range_c(1,:));
legend([hline2 hline1 hlined],'Estimates          ','True tracks','Measurements');

subplot(212); xlabel('Time'); ylabel('y-coordinate (m)');
set(gca, 'XLim',[1 truth.K]); set(gca, 'YLim',model.range_c(2,:));
%legend([yhline2 yhline1 yhlined],'Estimates          ','True tracks','Measurements');

%plot error - OSPA
ospa_vals= zeros(truth.K,3);
ospa_c= 100;
ospa_p= 1;
for k=1:meas.K
    [ospa_vals(k,1), ospa_vals(k,2), ospa_vals(k,3)]= ospa_dist(get_comps(truth.X{k},[1 3]),get_comps(est.X{k},[1 3]),ospa_c,ospa_p);
end

figure; ospa= gcf; hold on;
subplot(3,1,1); plot(1:meas.K,ospa_vals(:,1),'k'); grid on; set(gca, 'XLim',[1 meas.K]); set(gca, 'YLim',[0 ospa_c]); ylabel('OSPA Dist');
subplot(3,1,2); plot(1:meas.K,ospa_vals(:,2),'k'); grid on; set(gca, 'XLim',[1 meas.K]); set(gca, 'YLim',[0 ospa_c]); ylabel('OSPA Loc');
subplot(3,1,3); plot(1:meas.K,ospa_vals(:,3),'k'); grid on; set(gca, 'XLim',[1 meas.K]); set(gca, 'YLim',[0 ospa_c]); ylabel('OSPA Card');
xlabel('Time');

%plot error - OSPA^(2)
order = 1;
cutoff = 100;
win_len= 10;

ospa2_cell = cell(1,length(win_len));
for i = 1:length(win_len)
    ospa2_cell{i} = compute_ospa2(X_track([1 3],:,:),Y_track([1 3],:,:),cutoff,order,win_len);
end

figure; ospa2= gcf; hold on;
windowlengthlabels = cell(1,length(win_len));
subplot(3,1,1);
for i = 1:length(win_len)
    plot(1:truth.K,ospa2_cell{i}(1,:),'k'); grid on; set(gca, 'XLim',[1 meas.K]); set(gca, 'YLim',[0 cutoff]); ylabel('OSPA$^{(2)}$ Dist','interpreter','latex');
    windowlengthlabels{i} = ['$L_w = ' int2str(win_len(i)) '$'];
end
legend(windowlengthlabels,'interpreter','latex');

subplot(3,1,2);
for i = 1:length(win_len)
    plot(1:truth.K,ospa2_cell{i}(2,:),'k'); grid on; set(gca, 'XLim',[1 meas.K]); set(gca, 'YLim',[0 cutoff]); ylabel('OSPA$^{(2)}$ Loc','interpreter','latex');
    windowlengthlabels{i} = ['$L_w = ' int2str(win_len(i)) '$'];
end

subplot(3,1,3);
for i = 1:length(win_len)
    plot(1:truth.K,ospa2_cell{i}(3,:),'k'); grid on; set(gca, 'XLim',[1 meas.K]); set(gca, 'YLim',[0 cutoff]); ylabel('OSPA$^{(2)}$ Card','interpreter','latex');
    windowlengthlabels{i} = ['$L_w = ' int2str(win_len(i)) '$'];
end
xlabel('Time','interpreter','latex');
  
  
%plot cardinality
figure; cardinality= gcf; 
subplot(2,1,1); box on; hold on;
stairs(1:meas.K,truth.N,'k'); 
plot(1:meas.K,est.N,'k.');

grid on;
legend(gca,'True','Estimated');
set(gca, 'XLim',[1 meas.K]); set(gca, 'YLim',[0 max(truth.N)+1]);
xlabel('Time'); ylabel('Cardinality');

%return
handles=[ truths tracking ospa ospa2 cardinality ];


function ca= makecolorarray(nlabels)
    lower= 0.1;
    upper= 0.9;
    rrr= rand(1,nlabels)*(upper-lower)+lower;
    ggg= rand(1,nlabels)*(upper-lower)+lower;
    bbb= rand(1,nlabels)*(upper-lower)+lower;
    ca.rgb= [rrr; ggg; bbb]';
    ca.lab= cell(nlabels,1);
    ca.cnt= 0;   
end

function idx= assigncolor(label)
    str= sprintf('%i*',label);
    tmp= strcmp(str,colorarray.lab);
    if any(tmp)
        idx= find(tmp);
    else
        colorarray.cnt= colorarray.cnt + 1;
        colorarray.lab{colorarray.cnt}= str;
        idx= colorarray.cnt;
    end
end

function count= countestlabels
    labelstack= [];
    for k=1:meas.K
        labelstack= [labelstack est.L{k}];
    end
    [c,~,~]= unique(labelstack','rows');
    count=size(c,1);
end

end

function [X_track,k_birth,k_death]= extract_tracks(X,track_list,total_tracks)

K= size(X,1); 
x_dim= size(X{K},1); 
k=K-1; while x_dim==0, x_dim= size(X{k},1); k= k-1; end
X_track= NaN(x_dim,K,total_tracks);
k_birth= zeros(total_tracks,1);
k_death= zeros(total_tracks,1);

max_idx= 0;
for k=1:K
    if ~isempty(X{k})
        X_track(:,k,track_list{k})= X{k};
    end
    if max(track_list{k})> max_idx %new target born?
        idx= find(track_list{k}> max_idx);
        k_birth(track_list{k}(idx))= k;
    end
    if ~isempty(track_list{k}), max_idx= max(track_list{k}); end
    k_death(track_list{k})= k;
end
end


function Xc= get_comps(X,c)

if isempty(X)
    Xc= [];
else
    Xc= X(c,:);
end
end