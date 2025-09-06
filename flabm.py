import time
from abm_module import AircraftCabin, MovementController, TransmissionModel
from fluent_module import FluentInterface, FluentInterfaceFallback

# ------------------------------
# Simulation Parameters
# ------------------------------
SIMULATION_TIME = 3600  # 1 hour in seconds
TIME_STEP = 1.0         # 1 second time steps
FLUENT_UPDATE_INTERVAL = 60  # Update CFD every 60 seconds
FLUENT_MESH_PATH = "aircabin.msh"  # Path to your mesh file

def run_simulation():
    # Initialize cabin and ABM components
    cabin = AircraftCabin()
    movement_controller = MovementController()
    
    # Initialize Fluent interface
    try:
        fluent_interface = FluentInterface(FLUENT_MESH_PATH)
        print("Successfully connected to ANSYS Fluent")
    except Exception as e:
        print(f"Failed to connect to Fluent: {e}")
        print("Using fallback concentration model")
        fluent_interface = None
    
    total_agents = len(cabin.agents)
    initial_infected = sum(1 for a in cabin.agents if a.infected)
    print(f"Starting simulation with {total_agents} agents in 3-3-3 layout")
    print(f"Front section: {cabin.front_rows} rows, Rear section: {cabin.rear_rows} rows")
    print(f"Initial infected count: {initial_infected}")
    
    # Run simulation
    try:
        for _ in range(int(SIMULATION_TIME / TIME_STEP)):
            # Update agent behaviors and movements
            for agent in cabin.agents:
                movement_controller.decide_next_action(agent, cabin)
                movement_controller.move_agent(agent, cabin)
            
            # Run Fluent simulation periodically
            if cabin.current_time % FLUENT_UPDATE_INTERVAL == 0:
                if fluent_interface:
                    try:
                        fluent_interface.update_contamination_sources(cabin.agents)
                        concentration_data = fluent_interface.run_simulation()
                        cabin.update_concentration_data(concentration_data)
                    except Exception as e:
                        print(f"Fluent simulation error: {e}")
                else:
                    # Use fallback model if Fluent isn't available
                    concentration_data = FluentInterfaceFallback.run_simulation(cabin)
                    cabin.update_concentration_data(concentration_data)
            
            # Update infections
            TransmissionModel.update_infections(cabin)
            
            # Update simulation time
            cabin.current_time += TIME_STEP
            
            # Print progress
            if cabin.current_time % 300 == 0:  # Every 5 minutes
                infected_count = sum(1 for a in cabin.agents if a.infected)
                front_infected = sum(1 for a in cabin.agents if a.infected and a.seat.section == "front")
                rear_infected = infected_count - front_infected
                print(f"Time: {cabin.current_time:.0f}s, Total Infected: {infected_count}/{total_agents}")
                print(f"  Front: {front_infected}, Rear: {rear_infected}")
    
    finally:
        # Clean up Fluent session
        if fluent_interface:
            fluent_interface.close()
    
    print("\nSimulation complete")
    print(f"Final infected count: {sum(1 for a in cabin.agents if a.infected)}/{total_agents}")
    print(f"Total surface contacts: {len(cabin.contact_events)}")

if __name__ == "__main__":
    run_simulation()
    