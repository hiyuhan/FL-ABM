# FL-ABM
A fluid-mediated agent-based model for infection transmission in airplane cabin


## Abstract

This project develops a **Fluid-Mediated Agent-Based Model (FL-ABM)** to quantify airborne pathogen spread in aircraft cabins. By integrating ANSYS Fluent (CFD for airflow) with agent-based modeling (ABM for passenger behavior), it identifies high-risk areas (e.g., near infected passengers, window seats, bathrooms) and evaluates cost-effective interventions.

## Prerequisites (What to Install)

### 1. Software



*   Python 3.8 or later

*   ANSYS Fluent 2022 R2 or later (for CFD simulations)

### 2. Python Libraries

Install required packages via command line:



```
pip install numpy ansys-fluent-core
```

### 3. Input File



*   A pre-generated Fluent mesh file (`.msh`) of the 3-3-3 aircraft cabin.

    *Note: Update the mesh file path in *`flabm.py`* (variable: *`FLUENT_MESH_PATH`*).*

## How to Run



1.  **Prepare Files**: Ensure the Fluent mesh file is in an accessible folder, and update its path in `flabm.py`.

2.  **Adjust Settings (Optional)**: Modify key parameters in `flabm.py` if needed (e.g., `SIMULATION_TIME` for duration, initial infection rate).

3.  **Start Simulation**:



```
python flabm.py
```

