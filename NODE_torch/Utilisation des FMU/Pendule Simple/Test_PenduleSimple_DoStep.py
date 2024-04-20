from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import numpy as np
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

class FMU:
    def __init__(self, fmu_filename, inputs, outputs):
        self.filename = fmu_filename

        self.model_description = read_model_description(fmu_filename, validate=False)

        # Valeurs de référence
        self.vrs = {}
        for variable in self.model_description.modelVariables:
            self.vrs[variable.name] = variable.valueReference

        # Entrée et sortie
        self.vrs_input = []
        for i in inputs:
            self.vrs_input.append(self.vrs[i])
        self.vrs_output = []
        for o in outputs:
            self.vrs_output.append(self.vrs[o])

        # Extraire la fmu
        self.unzipdir = extract(fmu_filename)
        
        self.fmu = FMU2Slave(guid=self.model_description.guid,
                             unzipDirectory=self.unzipdir,
                             modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                             instanceName='instance')
        
        # Initialisation de now_input à None
        self.now_input = None
    
    def initialize(self, start_time):
        # Initialisation
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=start_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

        self.passed_in = np.array([])
        self.passed_out = np.array([])

    def setInputs(self, inputs):
        self.fmu.setReal(self.vrs_input, inputs)
        self.now_input = inputs 

    def getOutputs(self):
        return self.fmu.getReal(self.vrs_output)

    def doStep(self, time, step_size):
        self.fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    
        self.passed_in = np.append(self.passed_in, self.now_input)
        outputs = self.getOutputs()
        self.passed_out = np.append(self.passed_out, outputs)
    
        return outputs

    def terminate(self):
        self.fmu.terminate()
        self.fmu.freeInstance()
        
        shutil.rmtree(self.unzipdir)

# Charger le FMU
fmu = FMU('PenduleSimple_2_sorties.fmu', inputs=['In1'], outputs=['Out1', 'Out2'])
fmu.initialize(start_time=0.0)

# Temps de simulation
start_time = 0.0
end_time = 10.0
step_size = 0.005
n_steps = int((end_time - start_time) / step_size)

# Simulation
outputs_theta = []  
outputs_omega = [] 
prev_theta = None  

for i in range(n_steps):
    t = start_time + i * step_size 
    # Définition de l'entrée du FMU
    fmu.setInputs([t])
    # Simulation
    output = fmu.doStep(time=float(t), step_size=step_size)
    theta, omega = output
    outputs_omega.append(omega)
    outputs_theta.append(theta)
    prev_theta = theta

# Terminer le FMU
fmu.terminate()

# Affichage
time_values = np.linspace(start_time, end_time, n_steps)
plt.plot(time_values, outputs_theta, label='Theta')
plt.plot(time_values, outputs_omega, label='Omega')  
plt.xlabel('Temps')
plt.ylabel('Valeurs')
plt.title('Angle Theta et Vitesse ang. Omega avec DoStep')
plt.legend()
plt.show()
