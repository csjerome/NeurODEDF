# NeurODEDF

Dans ce github, vous retrouverez :
- les différents codes d'Aphynity sur lesquels nous nous sommes appuyés lors de nos travaux
- un fichier regroupant nos différents compte-rendu tout au long de l'année
- un fichier avec certains des papiers que nous avons pu étudier
- un fichier sur le cas du pendule simple
- un fichier sur le cas intermédiaire du pont roulant

Cas du pendule simple (fichier pendule) :
Vous y retrouverez nos différents codes notamment sur la réalisation du réseau de neurones, les différents modèles que nous avons mis en oeuvre, leurs entrainements et l'implémentation du MPC sur le cas du pendule
Vous retrouverez notamment dans le code net.py la réalisation du réseau de neurones avec ces différents poids et biais, ou encore dans le code testpythonimplmpc.py l'implémentation du controle par MPC sur le pendule simulé sous Python

Cas du pont roulant (fichier NODE_torch) :
De même que pour le cas du pendule simple, vous y retrouverez les différents travaux que nous avons pu effectuer sur cet exemple visant à nous rapporcher du modèle de la chaudière.
Les différents codes et noms de cette partie ont été réalisé de manière similaire à ceux correspondant au pendule simple

Utilisation des FMU (fichier du même nom dans la partie NODE_torch) :
Vous retrouverez pour les deux cas précédents les codes et FMU nécessaires aux différentes simulations effectués dans le rapport final
Dans les codes, vous trouverez notamment le code Test-FMU.py qui permet de visualiser les sorties de la FMU générées sur OpenModelica, les FMU nécessaires à l'éxecution du code, le code Test_PID pour visualiser la commande du pendule avec un correcteur PID ainsi que le code Test_mpc.py pour asservir la sortie du pendule par un controleur MPC
Cela est fais de façon similaire pour le cas du pont roulant
