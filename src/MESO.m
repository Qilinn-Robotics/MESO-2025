%___________________________________________________________________%
%  Multi-Strategies Enhanced Snake Optimizer (MESO)                 %
%                                                                   %
%  source codes STABLE version 1.0                                  %
%                                                                   %
%  Developed in MATLAB R2023b                                       %
%                                                                   %
%  Author and programmer:  Qilin Li & Xin Weng                      %
%                                                                   %
%  E-Mail: qilin516@outlook.com                                     %
%___________________________________________________________________%

%% STABLE
function [fval,Xfood,gbest_t] = MESO(N,T,lb,ub,dim,fobj)
%initial
alpha=0.05;
beta=0.15;

vec_flag=[1,-1];
Threshold=0.25;
Thresold2= 0.6;
C1=0.25; % 0.5
C2=0.1; % 0.05
C3=3.0; % 2
X=lb+rand(N,dim)*(ub-lb);

%Diving the swarm into two equal groups males and females
Nm=round(N/2);%eq.(3)
Nf=N-Nm;
Xm=X(1:Nm,:);
Xf=X(Nm+1:N,:);
% malloc
Xnewm=zeros(Nm,dim);
Xnewf=zeros(Nm,dim);
fitness=zeros(N,1);

for i=1:N % eval fitness
    fitness(i)=feval(fobj,X(i,:));
end
entropy_value=entropy_eval(fitness);

fitness_m=fitness(1:Nm);
fitness_f=fitness(Nm+1:N);
[fitnessBest_m,mBest_index] = min(fitness_m);
[fitnessBest_f,fBest_index] = min(fitness_f);
[fitnessWorst_m,mWorst_index] = max(fitness_m);
[fitnessWorst_f,fWorst_index] = max(fitness_f);
Xbest_m=Xm(mBest_index,:);
Xbest_f=Xf(fBest_index,:);
Xworst_m=Xm(mWorst_index,:);
Xworst_f=Xf(fWorst_index,:);

[GYbest, gbest] = min(fitness);
Xfood = X(gbest,:);

%% main loop
for t = 1:T
    Temp_o=exp(-((t)/T));  %eq.(4)
    Q_o=C1*exp(((t-T)/(T)));%eq.(5)
    % entropy improvement
    Q = Q_o * (1 + beta * entropy_value);%eq.(26)
    Temp = Temp_o * (1-alpha * entropy_value);%eq.(27)
    if Q>1
        Q=1;
    end
    
    RL=levy(N,dim,1.5); %eq.(25)
    % Exploration Phase (no Food)
    if Q<Threshold
        for i=1:Nm
            for j=1:1:dim
                rand_leader_index = floor(Nm*rand()+1);
                X_randm = Xm(rand_leader_index, :);
                flag_index = floor(2*rand()+1);
                Flag=vec_flag(flag_index);
                Am=exp(-fitness_m(rand_leader_index)/(fitness_m(i)+eps));%eq.(8)
                Xnewm_o(i,j)=X_randm(j)+Flag*C2*Am*((ub-lb)*rand+lb);%eq.(6)

                % Roulette-m
                total_fit = sum(1./(fitness_f + eps));
                prob_selection = (1./(fitness_f + eps)) / total_fit;%eq.(22)
                cum_prob = cumsum(prob_selection);
                r = rand();
                selected_index = find(cum_prob >= r, 1, 'first');
                X_rou_m = Xm(selected_index, :);
                Xnewm_n(i,j)=Xm(i,j)+C2*RL(i,j)*abs(X_rou_m(j)-Xm(i,j));%eq.(23)
            end
            if fobj(Xnewm_n(i,:))<fobj(Xnewm_o(i,:))
                Xnewm(i,:) = Xnewm_n(i,:);
            else
                Xnewm(i,:) = Xnewm_o(i,:);
            end
        end
        for i=1:Nf
            for j=1:1:dim
                rand_leader_index = floor(Nf*rand()+1);
                X_randf = Xf(rand_leader_index, :);
                flag_index = floor(2*rand()+1);
                Flag=vec_flag(flag_index);
                Af=exp(-fitness_f(rand_leader_index)/(fitness_f(i)+eps));%eq.(9)
                Xnewf_o(i,j)=X_randf(j)+Flag*C2*Af*((ub-lb)*rand+lb);%eq.(7)

                %Roulette-f
                total_fit = sum(1./(fitness_f + eps));
                prob_selection = (1./(fitness_f + eps)) / total_fit;%eq.(22)
                cum_prob = cumsum(prob_selection);
                r = rand();
                selected_index = find(cum_prob >= r, 1, 'first');
                X_rou_f = Xf(selected_index, :);
                Xnewf_n(i,j)=Xf(i,j)+C2*RL(i,j)*abs(X_rou_f(j)-Xf(i,j));%eq.(24)
            end
            if fobj(Xnewf_n(i,:))<fobj(Xnewf_o(i,:))
                Xnewf(i,:) = Xnewf_n(i,:);
            else
                Xnewf(i,:) = Xnewf_o(i,:);
            end
        end
    else %Exploitation Phase (Food Exists)
        if Temp>Thresold2  %hot
            for i=1:Nm
                flag_index = floor(2*rand()+1);
                Flag=vec_flag(flag_index);
                for j=1:1:dim
                    Xnewm(i,j)=Xfood(j)+C3*Flag*rand*(Xfood(j)-Xm(i,j));%eq.(10)
                end
            end
            for i=1:Nf
                flag_index = floor(2*rand()+1);
                Flag=vec_flag(flag_index);
                for j=1:1:dim
                    Xnewf(i,j)=Xfood(j)+Flag*C3*rand*(Xfood(j)-Xf(i,j));%eq.(11)
                end
            end
        else %cold
            if rand>0.6 %fight
                for i=1:Nm
                    for j=1:1:dim
                        FM=exp(-(fitnessBest_f)/(fitness_m(i)+eps));%eq.(14)
                        Xnewm(i,j)=Xm(i,j) +C3*FM*rand*(Xbest_f(j)-Xm(i,j));%eq.(12)

                    end
                end
                for i=1:Nf
                    for j=1:1:dim
                        FF=exp(-(fitnessBest_m)/(fitness_f(i)+eps));%eq.(15)
                        Xnewf(i,j)=Xf(i,j)+C3*FF*rand*(Xbest_m(j)-Xf(i,j));%eq.(13)
                    end
                end
            else%mating
                for i=1:Nm
                    for j=1:1:dim
                        Mm=exp(-fitness_f(i)/(fitness_m(i)+eps));%eq.(18)
                        Xnewm(i,j)=Xm(i,j)+C3*rand*Mm*(Xf(i,j)-Xm(i,j));%eq.(16)
                    end
                end
                for i=1:Nf
                    for j=1:1:dim
                        Mf=exp(-fitness_m(i)/(fitness_f(i)+eps));%eq.(19)
                        Xnewf(i,j)=Xf(i,j)+C3*rand*Mf*(Xm(i,j)-Xf(i,j));%eq.(17)
                    end
                end
                flag_index = floor(2*rand()+1);
                egg=vec_flag(flag_index);
                if egg==1
                    %% IEPD strategy
                    X=[Xnewm;Xnewf];
                    fit1 = zeros(N,dim);
                    for i=1:N
                        fit1(i)=feval(fobj,X(i,:));
                    end
                    [sorted_f,I1]=sort(fit1);
                    a=ceil(N/2);
                    freq=1/dim;
                    epsilon=1/2*(sin(2*pi*freq*t+pi)*(t/T)+1); %eq.(33)
                    for i=1:a
                        z1=I1(i);
                        dx = randperm(N,2);
                        if dx(1)== i
                            r_x=dx(2);
                        else
                            r_x=dx(1);
                        end
                        Eegg = X(I1(i),:) + epsilon*(Xfood - X(r_x,:));%eq.(30)
                        fit_mson=feval(fobj,Eegg);
                        if fit_mson < sorted_f(i)
                            Xnewm(z1,:)= Eegg;
                        end
                    end
                    for i=a+1:N
                        z1 = I1(i);
                        Iegg = Xfood + sign(rand-0.50)*(lb+rand*(ub-lb)*ones(1,dim));%eq.(31)
                        fit2 = feval(fobj, Iegg);
                        if fit2 < fit1(z1)
                            X(z1,:) = Iegg;
                        else
                            X(z1,:) = lb + rand(1,dim).*(ub-lb);
                        end
                    end
                    for i=1:Nm
                        Xnewm(i,:)=X(i,:);
                        Xnewf(i,:)=X(i+Nm,:);
                    end
                end
            end
        end
    end
    %% Constraints process
    for j=1:Nm
        Flag4ub=Xnewm(j,:)>ub;
        Flag4lb=Xnewm(j,:)<lb;
        Xnewm(j,:)=(Xnewm(j,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        y = feval(fobj,Xnewm(j,:));
        if y<fitness_m(j)
            fitness_m(j)=y;
            Xm(j,:)= Xnewm(j,:);
        end
    end
    for j=1:Nf
        Flag4ub=Xnewf(j,:)>ub;
        Flag4lb=Xnewf(j,:)<lb;
        Xnewf(j,:)=(Xnewf(j,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        y = feval(fobj,Xnewf(j,:));
        if y<fitness_f(j)
            fitness_f(j)=y;
            Xf(j,:)= Xnewf(j,:);
        end
    end

    fitness=[fitness_m',fitness_f'];
    [GYbest_,~]=min(fitness);
    entropy_value=entropy_eval(fitness);
    [Ybest1,gbest1] = min(fitness_m);   
    [Ybest2,gbest2] = min(fitness_f);   

    if Ybest1<fitnessBest_m
        Xbest_m = Xm(gbest1,:);
        fitnessBest_m=Ybest1;
    end
    if Ybest2<fitnessBest_f
        Xbest_f = Xf(gbest2,:);
        fitnessBest_f=Ybest2;
    end
    if Ybest1<Ybest2
        gbest_t(t)=Ybest1;
    else
        gbest_t(t)=Ybest2;
    end
    if fitnessBest_m<fitnessBest_f 
        GYbest=fitnessBest_m;
        Xfood=Xbest_m;
    else
        GYbest=fitnessBest_f;
        Xfood=Xbest_f;
    end
end
fval = GYbest;
end

function [z] = levy(n,m,beta)

num = gamma(1+beta)*sin(pi*beta/2);

den = gamma((1+beta)/2)*beta*2^((beta-1)/2);

sigma_u = (num/den)^(1/beta);

u = random('Normal',0,sigma_u,n,m);

v = random('Normal',0,1,n,m);

z =u./(abs(v).^(1/beta));

end

function entropy_value = entropy_eval(fitness)

fitness(fitness == 0) = 1e-10;

total_fitness = sum(fitness);

relative_fitness = fitness / total_fitness;

entropy_value = -sum(relative_fitness .* log2(relative_fitness));%eq.(28)
end


