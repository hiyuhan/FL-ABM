import numpy as np
import time
import ansys.fluent.core as pyfluent
from abm_module import AircraftCabin, Agent, ConcentrationData

# ------------------------------
# Fluent Integration Class
# ------------------------------
class FluentInterface:
    def __init__(self, mesh_path: str):
        self.session = None
        self.mesh_path = mesh_path
        self.initialize_fluent()
        
    def initialize_fluent(self):
        """Initialize Fluent session with 3-3-3 cabin geometry"""
        # Start Fluent in 2D mode with double precision
        self.session = pyfluent.launch_fluent(version="2d", precision="double", show_gui=False)
        
        # Load mesh for 3-3-3 cabin
        self.session.file.read_mesh(file_name=self.mesh_path)
        
        # Define fluid properties (air)
        self.session.setup.models.viscous.model = "k-epsilon"
        self.session.setup.materials.fluid["air"].copy_from("air")
        
        # Enable species transport for contamination tracking
        self.session.setup.models.species.model = "species-transport"
        self.session.setup.models.species.species = "contaminant"
        
        # Set boundary conditions - updated for 3-3-3 layout
        self.set_boundary_conditions()
        
        # Initialize solution
        self.session.solution.initialization.hybrid_initialize()
        
    def set_boundary_conditions(self):
        """Set up boundary conditions for 3-3-3 cabin with proper ventilation"""
        # Define inlet (HVAC) - multiple inlets for larger cabin
        if "inlet_front" in self.session.setup.boundary_conditions.pressure_inlet:
            self.session.setup.boundary_conditions.pressure_inlet["inlet_front"].momentum.velocity_specification_method = "magnitude"
            self.session.setup.boundary_conditions.pressure_inlet["inlet_front"].momentum.magnitude = 0.12  # m/s
        
        if "inlet_rear" in self.session.setup.boundary_conditions.pressure_inlet:
            self.session.setup.boundary_conditions.pressure_inlet["inlet_rear"].momentum.velocity_specification_method = "magnitude"
            self.session.setup.boundary_conditions.pressure_inlet["inlet_rear"].momentum.magnitude = 0.12  # m/s
        
        # Define outlet
        if "outlet" in self.session.setup.boundary_conditions.pressure_outlet:
            self.session.setup.boundary_conditions.pressure_outlet["outlet"].pressure.gauge_pressure = 0  # Pa
        
        # Define walls and seats
        for boundary in self.session.setup.boundary_conditions.wall.get_object_names():
            self.session.setup.boundary_conditions.wall[boundary].thermal.thermal_condition = "temperature"
            self.session.setup.boundary_conditions.wall[boundary].thermal.temperature = 293.15  # 20°C
        
    def update_contamination_sources(self, infected_agents: list[Agent]):
        """Update contamination sources based on current infected agent positions"""
        # Remove previous point sources
        for source in self.session.setup.boundary_conditions.point_source.get_object_names():
            self.session.setup.boundary_conditions.point_source[source].delete()
        
        # Create new point sources for infected agents
        for i, agent in enumerate(infected_agents):
            if agent.infected:
                x, y = agent.position
                source_name = f"infected_source_{i}"
                self.session.setup.boundary_conditions.create_point_source(
                    name=source_name,
                    zone_id=i+1000,  # Unique zone ID
                    location=[x, y, 0]  # z=0 for 2D
                )
                # Set contamination emission rate
                self.session.setup.boundary_conditions.point_source[source_name].sources.species = 1e-8  # kg/s
    
    def run_simulation(self, num_iterations: int = 50) -> ConcentrationData:
        """Run Fluent simulation and return concentration data"""
        self.session.solution.run_calculation.iterate(iter_count=num_iterations)
        
        # Get mesh coordinates
        x_coords = self.session.results.graphics.mesh.x_coordinates()
        y_coords = self.session.results.graphics.mesh.y_coordinates()
        
        # Get concentration data
        concentration = self.session.results.graphics.contour.get_data(
            field="species-mass-fraction",
            species="contaminant"
        )
        
        return ConcentrationData(
            timestamp=time.time(),
            grid=(x_coords, y_coords),
            concentration=concentration
        )
    
    def close(self):
        """Close Fluent session"""
        if self.session:
            self.session.exit()

# ------------------------------
# Fallback Concentration Model
# ------------------------------
class FluentInterfaceFallback:
    @staticmethod
    def run_simulation(cabin: AircraftCabin) -> ConcentrationData:
        """Simple Gaussian distribution model for when Fluent isn't available"""
        # Create grid matching cabin dimensions
        x_grid = np.linspace(0, cabin.cabin_length, 50)
        y_grid = np.linspace(0, cabin.cabin_width, 20)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Initialize concentration field
        concentration = np.zeros_like(X)
        
        # Add contamination sources from infected agents
        for agent in cabin.agents:
            if agent.infected:
                ax, ay = agent.position
                # Gaussian distribution around infected agent
                concentration += 10 * np.exp(-0.5 * ((X - ax)/5)**2 - 0.5 * ((Y - ay)/2)** 2)
        
        return ConcentrationData(
            timestamp=cabin.current_time,
            grid=(x_grid, y_grid),
            concentration=concentration
        )
    