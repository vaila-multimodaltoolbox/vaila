function [elipse_area,SumAreasContorno1,areaelipse1,angpca1]=copanalise(dat)
% Rotina criada por Juliana Exel Santana e Felipe Arruda Moura e Paulo Santiago, finalizada
% em 02/04/2011.
% Essa rotina faz uma an·lise dos dados das coordenadas bidimensionais do
% centro de press„o na plataforma.
% Essa È a rotina central, que deve ser rodada.
% Para que essa rotina rode, È necess·rio ter na mesma pasta outras 3 
% rotinas: grafeigcop, contornocop, Contour2AreaII.
% O argumento dat È o arquivo dos dados bididimensionais do COP, em trÍs
% tentativas.
%
% O programa perguntar· 'Qual % do conjunto de dados (Z) vocÍ deseja
% representar?(Exemplo:90): '. Ao responder essa pergunta, a rotina
% calcular· a porcentagem referente ‡ altura m·xima encontrada na matriz Z.
% A partir disso, far· os c·lculos de ·reas e centrÛides dos contornos
% referentes ‡ essa porcentagem da altura m·xima encontrada em Z.
%
% DefiniÁ„o dos arquivos de saÌda:
% areaelipse1,areaelipse2,areaelipse3: Nos dados bidimensionais, foi
% realizada a PCA (centrada na media +- desvio devio padr„o) e calculada a
% area da elipse de confianÁa. Representam as ·reas da elipse nas
% tentativas 1, 2 e 3, respectivamente.
%
% angpca1,angpca1,angpca1: Angulo da primeira componente com o eixo "x". O
% ‚ngulo varia de -90 a 90 graus. Representam os angulos nas tentativas 1, 
% 2 e 3, respectivamente.
%
% SumAreasContorno1,SumAreasContorno2,SumAreasContorno3: somatÛria das
% ·reas dos contornos dos mapas topolÛgicos.

datf = filtbutter(dat,10,100);
%areaelipse2,angpca2,SumAreasContorno2,areaelipse3,angpca3,SumAreasContorno3

%%%% Define o tamanho da plataforma, onde se pretende fazer a an·lise. Caso o COP varie muito, pode aumentar esses valores. 
maxx=25;
maxy=25;
grid=1;

%%%% Organiza os dados em cada tentativa;
tentativa1 = 2.54*[-1*datf(:,1) datf(:,2)];


%%% Define o tamanho do grid que È feito na plataforma e calcula quantos
%%% instantes de tempo o COP esteve inserido em cada "quadrado" do grid.
xi=[maxx*-1:grid:maxx]';
yi=[maxy*-1:grid:maxy]';


j=1;
i=1;
    while i<=size(xi,1)-1;
        while j<=size(yi,1)-1;
              Z1(i,j)=size(find(tentativa1(:,1)>=xi(i) & tentativa1(:,1)<xi(i+1) & tentativa1(:,2)>=yi(j) & tentativa1(:,2)<yi(j+1)),1);
              j=j+1;
        end
            i=i+1;
            j=1;
    end

    
%%% Suaviza o mapa de superfÌcie 
for j=1:size(Z1,2);
    Zbb1(:,j)=interp1(1:size(Z1,1),Z1(:,j),linspace(1,size(Z1,1),100));
end
for j=1:size(Zbb1,1)
    Zx1(j,:)=interp1(1:size(Z1,1),Zbb1(j,:),linspace(1,size(Z1,1),100));
end


%%% PCA E CONTORNO TOPOGR¡FICO
% [ave1,sco1,ava1]=princomp(tentativa1);
% sava2t1=sqrt(ava1(2,1));
% sava1t1=sqrt(ava1(1,1));
% 
% 
% [ave2,sco2,ava2]=princomp(tentativa2);
% sava2t2=sqrt(ava2(2,1));
% sava1t2=sqrt(ava2(1,1));
% 
% [ave3,sco3,ava3]=princomp(tentativa3);
% sava2t3=sqrt(ava3(2,1));
% sava1t3=sqrt(ava3(1,1));

% Encontrando o nÌvel do contorno no qual se deseja calcular a ·rea e o
% centrÛide:
quest1 = input('Qual % do conjunto de dados (Z) vocÍ deseja representar?(Exemplo:90): ');

% Tentativa 1: GR¡FICO DO COP NA PLATAFORMA
% PCA:
y=linspace(maxx*-1,maxx,100)';
figure
subplot(1,3,1);
plot(tentativa1(:,1),tentativa1(:,2));
hold on
plot(tentativa1(1,1),tentativa1(1,2),'or');
plot(tentativa1(end,1),tentativa1(end,2),'sg');
daspect([1 1 1])
xlabel('Desloc. mÈdio-lateral (cm)');
ylabel('Desloc. antero-posterior (cm)');
xlim([maxx*-1 maxx])
ylim([maxx*-1 maxx])
elipse_area = ellipse_cop(tentativa1(:,1),tentativa1(:,2),'show',.95);


% % % GR¡FICO DE CALOR DO COP NA PLATAFORMA
subplot(1,3,2)
surf(y',y',Zx1','EdgeColor','none')
daspect([1 1 1])
shading interp
hold on
view(0,90)

[areaelipse1,angpca1]=grafeigcop(tentativa1);
xlabel('Desloc. mÈdio-lateral (cm)');
ylabel('Desloc. antero-posterior (cm)');
title(['¡rea da elipse = ',num2str(areaelipse1,'%6.3f'),' cm^2','   ¬ngulo da 1™ CP: ',num2str(angpca1,'%6.1f'),' graus'])
xlim([maxx*-1 maxx])
ylim([maxx*-1 maxx])
ellipse_cop_ngraf(tentativa1(:,1),tentativa1(:,2),'show',.95);



% Contorno Topogr·fico:
subplot(1,3,3)
% figure
[Centroid1,AreasContorno1,SumAreasContorno1]=contornocop(Zx1',quest1);
hold on
plot(Centroid1(1,:)',Centroid1(2,:)','rx')
daspect([1 1 1])
xlabel('Desloc. mÈdio-lateral (cm)');
ylabel('Desloc. antero-posterior (cm)');
title(['SomatÛria das ·reas das regiıes do contorno = ',num2str(SumAreasContorno1,'%6.3f'),' cm^2']);
legend('Contorno','CentrÛides');
view(0,90)
xlim([maxx*-1 maxx])
ylim([maxx*-1 maxx])
ellipse_cop_ngraf(tentativa1(:,1),tentativa1(:,2),'show',.95);



%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%





% FunÁ„o criada por Juliana Exel Santana e Felipe A. Moura em 29/03/2011.
% Esta rotina retorna o gr·fico dos contornos topogr·ficos do nÌvel
% desejado, juntamente com os centrÛides, ·reas de cada contorno e
% somatÛria das ·reas que formam o contorno no nÌvel indicado.
% Como argumento de entrada, carregue a "Z" do seu conjunto de dados. Essa
% matriz pode ser interpretada como as alturas referentes ao plano x-y.
% O programa perguntar· 'Qual % do conjunto de dados (Z) vocÍ deseja
% representar?(Exemplo:90): '. Ao responder essa pergunta, a rotina
% calcular· a porcentagem referente ‡ altura m·xima encontrada na matriz Z.
% A partir disso, far· os c·lculos de ·reas e centrÛides dos contornos
% referentes ‡ essa porcentagem da altura m·xima encontrada em Z.

function [Centroid,AreasContorno,SumAreasContorno]=contornocop(Z,quest1)

maxx=25;
% Plotando o mapa de calor:
y=linspace(maxx*-1,maxx,100)';
% figure
% subplot(1,2,1)
% surf(y,y,Z,'EdgeColor','none')
% shading interp
% hold on
% view(0,90)
% daspect([1 1 1])

%%%%%%%%%%%%% Fazendo os contornos topogr·ficos da matriz Z %%%%%%%%%%%%%%%

% Encontrando o nÌvel do contorno no qual se deseja calcular a ·rea e o
% centrÛide:
% quest1 = input('Qual % do conjunto de dados (Z) vocÍ deseja representar?(Exemplo:90): ');
quest1 = 100-quest1;
porcentagem = (max(max(Z))*quest1)/100;

% Plotando o contorno ao lado do mapa de calor:
% subplot(1,2,2)

% Contorno no nÌvel escolhido:
[c,g] = contour(y,y,Z,porcentagem,'b');

nC=length(c);
cc=1;j=1;

% Separando a matriz c em partes que contÈm as informaÁıes de cada
% contorno: 
while cc<nC
    ix(1:2,j)=c(1:2,cc);
    cvec_start(j)=cc+1;
    cc=cc+ix(2,j)+1;
    j=j+1;
end

cnovo=[];

% Encontrando "buracos" dentro dos contornos referentes ao nÌvel escolhido:
for j=1:size(ix,2)
    xc1=c(1,cvec_start(j):cvec_start(j)+ix(2,j)-1);
    yc1=c(2,cvec_start(j):cvec_start(j)+ix(2,j)-1);
    
    for h=1:size(ix,2)
    xc2=c(1,cvec_start(h):cvec_start(h)+ix(2,h)-1);
    yc2=c(2,cvec_start(h):cvec_start(h)+ix(2,h)-1);
    
    [in]= inpolygon(xc1,yc1,xc2,yc2);
    
    if in==0
        cvalido(j,h)=0;
        
        else cvalido(j,h)=1;
    end
    end

% Separando os contornos dos "buracos" encontrados para que sejam
% calculadas as ·reas e os centrÛides somente das regiıes pertencentes aos
% contornos:
    if sum(cvalido(j,:))==1;
        dados=[xc1;yc1];
        cnovo(:,size(cnovo,2)+1:size(xc1,2)+1+size(cnovo,2))=[ix(:,j) dados];
    end
        
end
            
% Para conferir se est„o sendo selecionados os contornos corretamente, ou
% seja, se est„o sendo excluÌdos os "buracos" dentro das regiıes dos
% contornos:
% nC=length(cnovo);
% cc=1;j=1;
% while cc<nC
%     ixn(1:2,j)=cnovo(1:2,cc);
%     cvec_start(j)=cc+1;
%     cc=cc+ixn(2,j)+1;
%     j=j+1;
% end


% Chamando a rotina que faz o c·lculo das ·reas das regiıes dos contornos 
% e calcula os centrÛides no nÌvel desejado:
[AreasContorno,Centroid,IN] = Contour2AreaII(cnovo);

% Somando todas as ·reas referentes ao nÌvel do contorno desejado e
% plotando em gr·fico junto aos centrÛides:
SumAreasContorno = sum(AreasContorno);
% hold on
% plot(Centroid(1,:),Centroid(2,:),'rx')
% xlabel('Desloc. mÈdio-lateral (cm)');
% ylabel('Desloc. antero-posterior (cm)');
% title(['SomatÛria das ·reas das regiıes do contorno= ',num2str(SumAreasContornos,'%6.3f'),' cm^2']);
% legend('Contorno','CentrÛides');
% view(0,90)
% daspect([1 1 1])







%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%





% Rotina criada por Felipe Arruda Moura para realizar PCA em dados
% bidimensionais. Criada e alterada em 02/04/2011.
% A estrutura de coorda deve ser n x 2.
% Os argumentos de saÌda s„o ·rea da elipse e o angulo entre a primeira
% componentes principal e o eixo x.


function [area,ang]=grafeigcop(coorda)

%%% Calcula os autovalores e autovetores
[ave,sco,ava]=princomp([coorda(:,1) coorda(:,2)]);
sava2=sqrt(ava(2,1));
sava1=sqrt(ava(1,1));

ave1=ave(:,1);
ave2=ave(:,2);

%%% Calcula o valor de ·rea e ‚ngulo
area=(pi*sava1*sava2);

if (ave1(1,1)>0 & ave1(2,1)>0) 
ang=acosd(ave1(1));

elseif ave1(1,1)>0 & ave1(2,1)<0
    ang=acosd(ave1(1))*-1;
    
elseif ave1(1,1)<0 & ave1(2,1)<0
    ang=180-acosd(ave1(1));
 
elseif (ave1(1,1)<0 & ave1(2,1)>0)
    ang=(180-acosd(ave1(1)))*-1;
end


mx=mean(coorda(:,1));
my=mean(coorda(:,2));

%%% Faz a representaÁ„o das componentes principais com a elipse
xlim([-1 1])
ylim([-1 1])
daspect([1 1 1]);

plot3([mx;(-sava2*ave1(2))+mx],[my;(sava2*ave1(1))+my],[10000 10000],'w','LineWidth',2)
plot3([mx;(sava1*ave2(2))+mx],[my;(-sava1*ave2(1))+my],[10000 10000],'w','LineWidth',2)
plot3([mx;(-sava1*ave2(2))+mx],[my;(sava1*ave2(1))+my],[10000 10000],'w','LineWidth',2)
plot3([mx;(sava2*ave1(2))+mx],[my;(-sava2*ave1(1))+my],[10000 10000],'w','LineWidth',2)
% texto=text(mx-1,my-1.5,num2str(nj),'FontSize',12);
% set(texto,'Color','r');
% set(texto,'FontWeight','bold');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Criando elipse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ave(2,1)<=0;
   ave(1,1)=ave(1,1)*-1;
   % rotacao da elipse
   giro=-acos(ave(1)); %angulo de rotacao da elipse
   ex=sava1; %eixo x da elipse
ey=sava2; %eixo y da elipse
%desenho
x=(-ex):0.0001:(ex);
ysup = ey.*sqrt( 1-((x)/ex).^2 );
yinf = -ey.*sqrt( 1-((x)/ex).^2 );
%rotacao da elipse
x1= x.*cos(giro) + ysup.*sin(giro);
x2= x.*cos(-giro) + ysup.*sin(-giro);
ysup1= -x.*sin(giro) + ysup.*cos(giro);
yinf1= -x.*sin(giro) - ysup.*cos(giro);
z(1,1:size(x1,2))=1000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%elipse rotacionada
plot3(x1+mx,ysup1+my,z,'w','LineWidth',2)
plot3(x2+mx,yinf1+my,z,'w','LineWidth',2);




else giro=-acos(ave(1)); %angulo de rotacao da elipse
%definicao dos semi-eixos
ex=sava1; %eixo x da elipse
ey=sava2; %eixo y da elipse
%desenho
x=(-ex):0.0001:(ex);
ysup = ey.*sqrt( 1-((x)/ex).^2 );
yinf = -ey.*sqrt( 1-((x)/ex).^2 );
%rotacao da elipse
x1= x.*cos(giro) + ysup.*sin(giro);
x2= x.*cos(-giro) + ysup.*sin(-giro);
ysup1= -x.*sin(giro) + ysup.*cos(giro);
yinf1= -x.*sin(giro) - ysup.*cos(giro);
z(1,1:size(x1,2))=1000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%elipse rotacionada
plot3(x1+mx,ysup1+my,z,'w','LineWidth',2)
plot3(x2+mx,yinf1+my,z,'w','LineWidth',2);

end





%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%



function [Area,Centroid,IN]=Contour2AreaII(C)
% % Syntax: [Area,Centroid,IN]=Contour2Area(C)
% % Takes the contour argument C from matlabs function contourc
% % as produced by C=contour(x,y,z,...) and convert the contours
% % to closed polygons from where the areas are calculated. 
% % In addition the centroids (centre of mass) Cxy are calculated
% % and a matrix IN determining the parent/child relationship
% % between the contours (if polygon i is inside j then IN_ij=1, else=0).
% % For obscure contours NaN would be retrived, but are excluded in output.
% %
% % Created By: Per Sundqvist 2010-01-26, ABB/CRC, V‰sterÂs/Sweden.
%
% %--- Example ---
% [X,Y,Z] = PEAKS(50);
% figure(1), clf;
% %C=contourf(X,Y,Z,0.37+[0 0]);
% C=contourf(X,Y,Z,5);
% [Area,Centroid,IN]=Contour2Area(C);
% xc=Centroid(1,:);yc=Centroid(2,:);
% hold on;plot(xc,yc,'k*');
% Area
% IN

%--- find number of contours ---
nC=length(C);
cc=1;j=1;
while cc<nC
    ix(j)=C(2,cc);
    cvec_start(j)=cc+1;
    cc=cc+ix(j)+1;
    j=j+1;
end
%--- find areas Ac and centroid Cxy (special if contour goes outside) ---
for j=1:length(ix)
    xC=C(1,cvec_start(j):cvec_start(j)+ix(j)-1);
    yC=C(2,cvec_start(j):cvec_start(j)+ix(j)-1);
    if ~isempty(find(isnan(xC)))
        if length(xC)>1
            xC(find(isnan(xC)))=(xC(find(isnan(xC))-1)+xC(find(isnan(xC))+1))/2;
            yC(find(isnan(yC)))=(yC(find(isnan(yC))-1)+yC(find(isnan(yC))+1))/2;
        end
    end
    if length(xC)>1
        Ac(j)=polyarea(xC,yC);  % area
        %--- determine clockwise/reverse sign, s
        s=(xC(2)-xC(1))*(yC(3)-yC(2))-(xC(3)-xC(2))*(yC(2)-yC(1));
        s=s/abs(s);
        Cxy(:,j)=s*sum([(xC(1:end-1)+xC(2:end)).*(xC(1:end-1).*yC(2:end)-xC(2:end).*yC(1:end-1));...
                  (yC(1:end-1)+yC(2:end)).*(xC(1:end-1).*yC(2:end)-xC(2:end).*yC(1:end-1))]')'/...
                  6/Ac(j);  % centroid (centre of mass)
        nan0(j)=0;
    else
        Ac(j)=NaN;
        Cxy(:,j)=[NaN;NaN];
        nan0(j)=1;
    end
    %hold on;plot(xC,yC,'r.',Cxy(1,j),Cxy(2,j),'b*'); % plot polygons and centroids
end
%--- Remove NaN contours ---
nanix=find(nan0~=1);
Area=Ac(nanix);
Centroid=Cxy(:,nanix); %(*)

%--- Determine relationship between polygons (inside eachother?, parent/child) ---

% Alterado por Juliana Exel Santana em 29/03/2011:
% Quando essa rotina, em sua forma original, calcula os centrÛides, em alguns
% casos, as suas coordenadas s„o apresentadas na matriz Centroid(*) com
% sinais trocados. Quando plotados, esses centrÛides acabam acusando uma
% posiÁ„o errada. A alteraÁ„o foi feita para resolver esse problema. Ent„o,
% È gerada uma nova matriz Centroid(**).

% IN=zeros(length(nanix),length(nanix));
for i=1:length(nanix)
    i0=nanix(i);
    xC=C(1,cvec_start(i0):cvec_start(i0)+ix(i0)-1);
    yC=C(2,cvec_start(i0):cvec_start(i0)+ix(i0)-1);
    
%     for j=i+1:length(nanix)
%         j0=nanix(j);
%         IN(i,j)=inpolygon(C(1,cvec_start(j0)),C(2,cvec_start(j0)),xC,yC);
          IN(1,i)=inpolygon(Centroid(1,i),Centroid(2,i),xC,yC);
 
%     end
end

[l,c]=find(IN==0);

Centroid(:,c)=-(Centroid(:,c)); %(**)


























function [area,axes,angles,ellip]=ellipse_cop(x,y,show,p)
% ELLIPSE calculates an ellipse that fits the data using principal component analysis
% [area,axes,angles,ellip]=ellipse(x,y,'show',p)
% X and Y are vectors with same length
% SHOW is an optional parameter to plot the data and the ellipse.
% P is an optional parameter to set the desired confidence area of the
%  ellipse. E.g., for p=.95 (default value), 95% of the data will lie
%  inside the ellipse. Use p=.8535 if you want semi-axes of the ellipse with a
%  length of 1.96 standard deviations (95% confidence interval in each axis).
% The outputs are the area of the ellipse (p*100% of the samples lie inside of
%  the ellipse), the axis lengths (major axis first), the respective angles (in rad),  
%  and the ellipse data.

% Marcos Duarte mduarte@usp.br 1999-2003

if exist('p') & ~isempty(p) & p~=.95
    if exist('raylinv.m')==2
        %The problem here is to find the probability p of having data with the distance
        % given by sqrt(x.^2+y.^2), wich has a Rayleigh distribution, less than a
        % certain value (the boundary of the ellipse).
        invp = raylinv(p,2)/2; %Inverse of the Rayleigh cumulative distribution function.
    else
        warning('Statistics toolbox not available. Using the default value p=.95')
        p = .95; invp = 4.8955/2;
    end
else
    p = .95; invp = 4.8955/2;
    %p = .8535; invp = 1.96 % uncomment this line in case you don't have the stats toolbox and want this axis
end
V = cov(x,y);                           % covariance matrix

% 1st way:
[vec,val] = eig(V);                     % eigenvectors and eigenvalues of the covariance matrix
axes = invp*sqrt(svd(val));             % axes
angles = -atan2( vec(1,:),vec(2,:) );   % angles  
area = pi*prod(axes);                   % area

% 2nd way (in case you don't want to use EIG and SVD):
%axes(1)=(V(1,1)+V(2,2)+sqrt( (V(1,1)-V(2,2))^2+4*V(2,1)^2 ))/2;
%axes(2)=(V(1,1)+V(2,2)-sqrt( (V(1,1)-V(2,2))^2+4*V(2,1)^2 ))/2;
%angles=atan2( V(1,2),axes-V(2,2) );   % angles
%axes=invp*sqrt(axes);                 % axes
%area=pi*prod(axes);                   % area
%vec=[cos(angles(1)) -sin(angles(1)); sin(angles(1)) cos(angles(1))];
%val=([axes(1) 0; 0 axes(2)]/1.96).^2;

% ellipse data:
t = linspace(0,2*pi);
ellip = vec*invp*sqrt(val)*[cos(t); sin(t)] + repmat([mean(x);mean(y)],1,100);
ellip = ellip';
axes  = axes';

% plot:
if exist('show') & ~isempty(show)
   %p2=polyfit(x,y,1);
   %fit=polyval(p2,[min(x) max(x)]);
   m = [mean(x) mean(x); mean(y) mean(y)];
   ax = [cos(angles); sin(angles)].*[axes; axes] + m;
%    figure
%    axis image
   set(gca,'box','on')
   hold on
   plot(x(1),y(1),'^k',x(end),y(end),'vm')
   plot(ellip(:,1),ellip(:,2),'r','linewidth',2)
   plot([ax(1,:); 2*m(1,:)-ax(1,:)],[ax(2,:); 2*m(2,:)-ax(2,:)],'r--','linewidth',2)
%    axis([-1 1 -1 1])
% axis('auto') 
   %plot([min(x) max(x)],fit,'k','linewidth',2)
   hold off
   %legend('Data','Linear regression','Ellipse & axes',0)
%    xlabel('X')
%    ylabel('Y')
   title([num2str(p*100) '% CONFIDENCE ELLIPSE (area = ' num2str(area) ', angle = ' num2str(round(angles(1)*180/pi*10)/10) '^o)'])
end












function [area,axes,angles,ellip]=ellipse_cop_ngraf(x,y,show,p)
% ELLIPSE calculates an ellipse that fits the data using principal component analysis
% [area,axes,angles,ellip]=ellipse(x,y,'show',p)
% X and Y are vectors with same length
% SHOW is an optional parameter to plot the data and the ellipse.
% P is an optional parameter to set the desired confidence area of the
%  ellipse. E.g., for p=.95 (default value), 95% of the data will lie
%  inside the ellipse. Use p=.8535 if you want semi-axes of the ellipse with a
%  length of 1.96 standard deviations (95% confidence interval in each axis).
% The outputs are the area of the ellipse (p*100% of the samples lie inside of
%  the ellipse), the axis lengths (major axis first), the respective angles (in rad),  
%  and the ellipse data.

% Marcos Duarte mduarte@usp.br 1999-2003

if exist('p') & ~isempty(p) & p~=.95
    if exist('raylinv.m')==2
        %The problem here is to find the probability p of having data with the distance
        % given by sqrt(x.^2+y.^2), wich has a Rayleigh distribution, less than a
        % certain value (the boundary of the ellipse).
        invp = raylinv(p,2)/2; %Inverse of the Rayleigh cumulative distribution function.
    else
        warning('Statistics toolbox not available. Using the default value p=.95')
        p = .95; invp = 4.8955/2;
    end
else
    p = .95; invp = 4.8955/2;
    %p = .8535; invp = 1.96 % uncomment this line in case you don't have the stats toolbox and want this axis
end
V = cov(x,y);                           % covariance matrix

% 1st way:
[vec,val] = eig(V);                     % eigenvectors and eigenvalues of the covariance matrix
axes = invp*sqrt(svd(val));             % axes
angles = -atan2( vec(1,:),vec(2,:) );   % angles  
area = pi*prod(axes);                   % area

% 2nd way (in case you don't want to use EIG and SVD):
%axes(1)=(V(1,1)+V(2,2)+sqrt( (V(1,1)-V(2,2))^2+4*V(2,1)^2 ))/2;
%axes(2)=(V(1,1)+V(2,2)-sqrt( (V(1,1)-V(2,2))^2+4*V(2,1)^2 ))/2;
%angles=atan2( V(1,2),axes-V(2,2) );   % angles
%axes=invp*sqrt(axes);                 % axes
%area=pi*prod(axes);                   % area
%vec=[cos(angles(1)) -sin(angles(1)); sin(angles(1)) cos(angles(1))];
%val=([axes(1) 0; 0 axes(2)]/1.96).^2;

% ellipse data:
t = linspace(0,2*pi);
ellip = vec*invp*sqrt(val)*[cos(t); sin(t)] + repmat([mean(x);mean(y)],1,100);
ellip = ellip';
axes  = axes';

% plot:
if exist('show') & ~isempty(show)
   %p2=polyfit(x,y,1);
   %fit=polyval(p2,[min(x) max(x)]);
   m = [mean(x) mean(x); mean(y) mean(y)];
   ax = [cos(angles); sin(angles)].*[axes; axes] + m;
%    figure
%    axis image
   set(gca,'box','on')
   hold on
%    plot(x,y,'bo',x(1),y(1),'^k',x(end),y(end),'vm')
   plot(ellip(:,1),ellip(:,2),'r','linewidth',2)
   plot([ax(1,:); 2*m(1,:)-ax(1,:)],[ax(2,:); 2*m(2,:)-ax(2,:)],'r--','linewidth',2)
%    axis([-1 1 -1 1])
% axis('auto') 
   %plot([min(x) max(x)],fit,'k','linewidth',2)
   hold off
   %legend('Data','Linear regression','Ellipse & axes',0)
%    xlabel('X')
%    ylabel('Y')
%    title([num2str(p*100) '% CONFIDENCE ELLIPSE (area = ' num2str(area) ', angle = ' num2str(round(angles(1)*180/pi*10)/10) '^o)'])
end



















function [datf] = filtbutter(dat,fc,freq)
if nargin == 2; freq = 100;end

n=4; %ordem do filtro

wn=fc/(freq/2);   %frequencia de corte de

[b,a] = butter(n,wn); %definindo o tipo de filtro Butterworth

ncol = size(dat,2);

for i = 1:ncol;
datf(:,i) = filtfilt(b,a,dat(:,i));

end

% datf = [dat(:,1),datf(:,2:end)];

datf = datf(:,1:end);
