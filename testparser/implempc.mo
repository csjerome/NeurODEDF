model implempc

parameter Integer Nsignaux = 100, nstepcrossentropy = 10, nkeptsignals = 10, Horizon = 10 ; //
parameter Real q = 1, r = 1, dt = 0.1, tempssimutot = 10; //paramètre de pondération du coût et temps de la simulation


parameter Real[integer(tempssimutot/dt)] trajref = ones(integer(tempssimutot/dt)); //trajectoire de référence tout au long de la simulation
parameter Real[2] condinit={-1,0}; //condition initiale de l'état du pendule
parameter Real alpha = 0.1, omega0_square = 1 ;

Real[Nsignaux, Horizon] TestSignals "variable de stockage des consignes u test";
Real[Nsignaux,Horizon, 2] TestResults "variable de stockage des valeurs de l'état pour ces consignes";
Real[nkeptsignals, Horizon] KeptSignals "signaux de consigne avec le meilleur score";
Real[Nsignaux] CostList "liste des coûts associés à chaque signal u";
Real[2] statevector"variable pour stocker le vecteur d'état";
Real u "consigne que l'on va appliquer";
Integer nHorizon "horizon que l'on peut restreindre notamment pour la fin où la trajectoire n'est pas définie pour tout l'horizon";
Real[Horizon] mu = zeros(Horizon), sigma=ones(Horizon);
Real[integer(tempssimutot/dt)] theta, thetap;
Real[20] variabledemerde;




function modelependulenn "reçoit l'état position et vitesse et la consigne en n et renvoit l'état en n+1"
  input Real[2] statevector;
  input Real command;
  output Real[2] nextstatevector;
end modelependulenn;


function generatesignals // fonction qui génère Nsignaux de taile Horizon selon une normale de paramètre mu sigma selon la méthode de Box-Muller
  input Real[:] muf "moyenne de la normale pour chaque étape de la consigne";
  input Real[:] sigmaf"écart-type de la normale pour chaque étape de la consigne";
  input Integer Nfsignaux;
  input Integer nfHorizon;
  output Real[:, :] Z;
  protected
    Real[:, :] U1, U2 "listes de signaux générés selon une loi uniforme";
  import Modelica.Math.Random.Generators.Xorshift64star.random;
  algorithm
  for i in 1:Nfsignaux loop
    for j in 1:nfHorizon loop
      U1[i, j] := random({1,1});
      U2[i, j] := random({1,1});
      Z[i, j] := sigmaf[j]*sqrt(-2*log(U1[i, j]))*cos(2*3.1415*U2[i, j])+muf[j] "on prend 3.1415 comme valeur de pi"; 
    end for;
  end for;      
end generatesignals;


function cost "calcule le coût pour une suite de consigne par rapport à la trajectoire de référence"
  input Real[nfHorizon] U "vecteur de consigne";
  input Real[nfHorizon] Xexpected "le vecteur prédit par le système avec réseau de neurone";
  input Real[nfHorizon] Xreference "le vecteur trajectoire de référence";
  input Integer nfHorizon;
  input Real qf;
  input Real rf;
  output Real J;
  import Modelica.Math.Vectors.norm;
  algorithm
      J := 1/2*qf*norm(Xexpected-Xreference)+1/2*rf*norm(U);
end cost;


function potentialtrajectory
  input Real[nHorizon] U;
  input Real[2] initstate;
  output Real[nHorizon] Xexpected;
  algorithm
  Xexpected[1] := initstate;
  for i in 2:size(U,1) loop
    Xexpected[i] := modelependulenn(Xexpected[i-1],U[i-1]);
  end for;
end potentialtrajectory;


function keepbest "garde les meilleurs signaux en fonction de leur coût"  
  input Real[Nfsignaux, nfHorizon] Signalstosort;
  input Real[Nfsignaux] costlist;
  input Integer Nfsignaux;
  input Integer nfHorizon;
  input Integer nfkeptsignals;
  output Real[nfkeptsignals, nfHorizon] Bettersignals;
  protected
    Real[Nfsignaux] costlistsorted;
    Integer[Nfsignaux] indexlist;
  import Modelica.Math.Vectors.sort;
  algorithm
    (costlistsorted, indexlist) := sort(costlist); //vérifier s'il faut pas mettre Vectors.sort(costlist)
    for i in 1:nfkeptsignals loop
      for j in 1:nfHorizon loop
        Bettersignals[i, j] := Signalstosort[indexlist[i], j];
      end for;
    end for;
end keepbest;


function average
  input Integer nin "Number of inputs";
  input Real u[nin] "Input vector";
  output Real y "Result"; 
  algorithm
    y := sum(u)/nin;
end average ;
function standarddeviation 
  input Integer nin "Number of inputs";
  input Real u[nin] "Input vector";
  output Real y "Result";
  algorithm
    y := (u[1]-average(nin,u))*(u[1]-average(nin,u));
    for i in 2:nin loop
      y := y + (u[i]-average(nin,u))*(u[i]-average(nin,u));
    end for;
    y := sqrt(y); 
end standarddeviation;


function modelependule
  input Real[2]positionandspeed;
  input Real consigne;
  input Real dtf;
  input Real fomega0_square;
  input Real falpha;
  output Real[2] newpositionandspeed;
  algorithm
    newpositionandspeed[1] := positionandspeed[1] + dtf*positionandspeed[2];
    newpositionandspeed[2] := positionandspeed[2] + dtf*(-fomega0_square*sin(positionandspeed[1]) - falpha * positionandspeed[2]);
end modelependule;



algorithm
  statevector := condinit;
  nHorizon := Horizon;
  for i in 1:integer(tempssimutot/dt) loop
    if integer(tempssimutot/dt) - i < Horizon then
      nHorizon := integer(tempssimutot/dt) - i;
    end if;
    for j in 1:size(TestSignals, 1) loop
      TestResults[j, 1] := statevector ;
    end for;
    for l in 1:nstepcrossentropy loop
      TestSignals := generatesignals(mu, sigma, Nsignaux, nHorizon);
      for j in 1:size(TestSignals, 1) loop
        for k in 2:nHorizon loop
          TestResults[j, k] := modelependule(TestResults[j, k-1], TestSignals[j, k-1], dt, omega0_square, alpha);
        end for;
      end for;
      for j in 1:Nsignaux loop
        CostList[j] := cost(TestSignals[j], TestResults[j, :, 1], trajref[i: i + nHorizon], nHorizon, q, r);
      end for;
      KeptSignals := keepbest(TestSignals, CostList, Nsignaux, nHorizon, nkeptsignals);
      for j in 1:size(KeptSignals,2) loop
        mu[j] := average(nkeptsignals, KeptSignals[j, : ]);
        sigma[j] := standarddeviation(nkeptsignals, KeptSignals[j, : ]);
      end for;
    end for;
  u := KeptSignals[1, 1];
  statevector := modelependule(statevector, u, dt, omega0_square, alpha);
  theta[i] := statevector[1];
  thetap[i] := statevector[2];  
  end for;



end implempc;
