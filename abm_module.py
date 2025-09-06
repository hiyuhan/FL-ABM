import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ------------------------------
# Simulation Parameters - ABM
# ------------------------------
CABIN_LENGTH = 40.0  # meters
CABIN_WIDTH = 6.0    # meters
FRONT_ROWS = 13      # 13 rows in front section
REAR_ROWS = 14       # 14 rows in rear section
SEATS_PER_ROW = 9    # 3-3-3 seating arrangement

# Bathroom positions: middle (between sections) and rear
MIDDLE_BATHROOM_POS = ( (FRONT_ROWS / (FRONT_ROWS + REAR_ROWS)) * CABIN_LENGTH, CABIN_WIDTH/2 )
BATHROOM_POSITIONS = [
    MIDDLE_BATHROOM_POS,  # Middle bathroom (between sections)
    (CABIN_LENGTH * 0.95, CABIN_WIDTH/2)  # Rear bathroom
]

# ------------------------------
# Data Structures
# ------------------------------
@dataclass
class Seat:
    row: int
    column: int
    position: Tuple[float, float]
    section: str  # 'front' or 'rear'
    occupied: bool = False

@dataclass
class Agent:
    id: int
    seat: Seat
    position: Tuple[float, float]
    destination: Tuple[float, float] = None
    moving: bool = False
    infected: bool = False
    infection_time: float = -1.0
    contact_count: int = 0

@dataclass
class ConcentrationData:
    timestamp: float
    grid: Tuple[np.ndarray, np.ndarray]
    concentration: np.ndarray

# ------------------------------
# Aircraft Cabin Environment with 3-3-3 Layout
# ------------------------------
class AircraftCabin:
    def __init__(self):
        self.front_rows = FRONT_ROWS
        self.rear_rows = REAR_ROWS
        self.seats = self._create_seating()
        self.agents = self._create_agents()
        self.concentration_data = None
        self.current_time = 0.0
        self.contact_events = []
        self.aisle_positions = self._get_aisle_positions()  # For reference
        
    def _create_seating(self) -> List[Seat]:
        """Create 3-3-3 seating with front (13 rows) and rear (14 rows) sections"""
        seats = []
        total_rows = FRONT_ROWS + REAR_ROWS
        seat_spacing = CABIN_LENGTH / (total_rows + 2)  # Extra space for bathrooms
        
        # Create front section (13 rows)
        for row in range(FRONT_ROWS):
            x_pos = (row + 1) * seat_spacing  # Start from front
            seats.extend(self._create_row_seats(row, x_pos, "front"))
        
        # Create rear section (14 rows) with spacing for middle bathroom
        for row in range(REAR_ROWS):
            # Add extra space for middle bathroom
            x_pos = (FRONT_ROWS + 1.5) * seat_spacing + (row + 1) * seat_spacing
            seats.extend(self._create_row_seats(row + FRONT_ROWS, x_pos, "rear"))
            
        return seats
    
    def _create_row_seats(self, row: int, x_pos: float, section: str) -> List[Seat]:
        """Create 9 seats (3-3-3) for a single row with two aisles"""
        row_seats = []
        
        # First section (3 seats)
        for col in range(3):
            y_pos = 0.8 + (col * 0.8)  # Leftmost section
            row_seats.append(Seat(
                row=row,
                column=col,
                position=(x_pos, y_pos),
                section=section
            ))
        
        # Second section (3 seats) - first aisle in between
        for col in range(3, 6):
            y_pos = 3.0 + ((col - 3) * 0.8)  # Middle section
            row_seats.append(Seat(
                row=row,
                column=col,
                position=(x_pos, y_pos),
                section=section
            ))
        
        # Third section (3 seats) - second aisle in between
        for col in range(6, 9):
            y_pos = 5.2 + ((col - 6) * 0.8)  # Rightmost section
            row_seats.append(Seat(
                row=row,
                column=col,
                position=(x_pos, y_pos),
                section=section
            ))
            
        return row_seats
    
    def _get_aisle_positions(self) -> List[float]:
        """Return Y positions of the two aisles in 3-3-3 layout"""
        return [2.4, 4.6]  # Between the three seat sections
    
    def _create_agents(self) -> List[Agent]:
        """Create agents and assign them to seats with 90% occupancy"""
        agents = []
        num_agents = int((FRONT_ROWS + REAR_ROWS) * SEATS_PER_ROW * 0.9)  # 90% occupancy
        
        # Randomly select seats to occupy
        occupied_seats = random.sample(self.seats, num_agents)
        for seat in occupied_seats:
            seat.occupied = True
        
        # Create agents for occupied seats
        for i, seat in enumerate(occupied_seats):
            agents.append(Agent(
                id=i,
                seat=seat,
                position=seat.position,
                infected=random.random() < 0.05  # 5% initial infection rate
            ))
        
        return agents
    
    def get_agent_by_id(self, agent_id: int) -> Agent:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def update_concentration_data(self, data: ConcentrationData):
        self.concentration_data = data
    
    def get_concentration_at_position(self, position: Tuple[float, float]) -> float:
        if self.concentration_data is None:
            return 0.0
            
        x, y = position
        x_grid, y_grid = self.concentration_data.grid
        conc = self.concentration_data.concentration
        
        x_idx = np.argmin(np.abs(x_grid - x))
        y_idx = np.argmin(np.abs(y_grid - y))
        
        return conc[x_idx, y_idx]

# ------------------------------
# Movement Controller with Aisle Awareness
# ------------------------------
class MovementController:
    @staticmethod
    def decide_next_action(agent: Agent, cabin: AircraftCabin) -> None:
        """Agents prefer nearest bathroom based on their section"""
        if not agent.moving:
            # Small probability to go to bathroom each second
            if random.random() < 0.0005:  # ~1-2 trips per hour
                # Prefer middle bathroom for front section, rear bathroom for rear section
                if agent.seat.section == "front" and random.random() < 0.7:
                    bathroom = BATHROOM_POSITIONS[0]  # Middle bathroom
                else:
                    # 30% chance front passengers use rear bathroom, 100% for rear passengers
                    bathroom = BATHROOM_POSITIONS[1] if random.random() < 0.3 or agent.seat.section == "rear" else BATHROOM_POSITIONS[0]
                
                agent.destination = bathroom
                agent.moving = True
        
        # If approaching bathroom, set next destination to seat
        elif agent.moving and np.linalg.norm(np.array(agent.position) - np.array(agent.destination)) < 0.5:
            agent.destination = agent.seat.position
        
        # If back at seat, stop moving
        elif agent.moving and np.linalg.norm(np.array(agent.position) - np.array(agent.destination)) < 0.3:
            agent.moving = False
            agent.destination = None
    
    @staticmethod
    def move_agent(agent: Agent, cabin: AircraftCabin, speed: float = 0.5) -> None:
        """Move agent towards destination, using aisles when possible"""
        if not agent.moving or agent.destination is None:
            return
            
        current_pos = np.array(agent.position)
        dest_pos = np.array(agent.destination)
        
        # Calculate path that uses aisles when moving long distances
        path = MovementController._calculate_path(current_pos, dest_pos, cabin)
        
        # Move along calculated path
        direction = path - current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            new_pos = current_pos + direction * speed * 1.0  # TIME_STEP is 1.0
            
            # Check for collisions with seats
            if MovementController._check_seat_collision(new_pos, cabin):
                agent.contact_count += 1
                cabin.contact_events.append({
                    'time': cabin.current_time,
                    'agent_id': agent.id,
                    'position': tuple(new_pos)
                })
                # Adjust position to avoid collision
                new_pos = MovementController._adjust_for_collision(new_pos, direction, cabin)
            
            agent.position = tuple(new_pos)
    
    @staticmethod
    def _calculate_path(start: np.ndarray, end: np.ndarray, cabin: AircraftCabin) -> np.ndarray:
        """Calculate path that uses aisles for longer movements"""
        # For short distances, go directly
        if np.linalg.norm(start - end) < 1.0:
            return end
            
        # For longer distances, use aisles as corridors
        start_x, start_y = start
        end_x, end_y = end
        
        # Get nearest aisle
        aisles = cabin.aisle_positions
        nearest_aisle = min(aisles, key=lambda a: abs(start_y - a))
        
        # If not already in an aisle, first move to nearest aisle
        if abs(start_y - nearest_aisle) > 0.3:
            return np.array([start_x, nearest_aisle])
        # Then move along aisle to correct x position
        elif abs(start_x - end_x) > 0.5:
            return np.array([end_x, nearest_aisle])
        # Then move from aisle to destination
        else:
            return end
    
    @staticmethod
    def _check_seat_collision(position: np.ndarray, cabin: AircraftCabin) -> bool:
        x, y = position
        for seat in cabin.seats:
            sx, sy = seat.position
            if np.sqrt((x - sx)**2 + (y - sy)** 2) < 0.4:  # Slightly smaller radius for 3-3-3
                return True
        return False
    
    @staticmethod
    def _adjust_for_collision(position: np.ndarray, direction: np.ndarray, cabin: AircraftCabin) -> np.ndarray:
        """Adjust position to stay in aisles when possible"""
        x, y = position
        aisles = cabin.aisle_positions
        
        # Try to adjust towards nearest aisle if not already in one
        if not any(abs(y - a) < 0.3 for a in aisles):
            nearest_aisle = min(aisles, key=lambda a: abs(y - a))
            return np.array([x, nearest_aisle])
        # Otherwise adjust perpendicular to movement direction
        else:
            perpendicular = np.array([-direction[1], direction[0]])
            return position + perpendicular * 0.1

# ------------------------------
# Transmission Model
# ------------------------------
class TransmissionModel:
    @staticmethod
    def update_infections(cabin: AircraftCabin) -> None:
        TransmissionModel._air_transmission(cabin)
        TransmissionModel._surface_transmission(cabin)
    
    @staticmethod
    def _air_transmission(cabin: AircraftCabin) -> None:
        if cabin.concentration_data is None:
            return
            
        for agent in cabin.agents:
            if agent.infected:
                continue
                
            conc = cabin.get_concentration_at_position(agent.position)
            infection_prob = 1 - np.exp(-0.1 * conc)  # Calibrate based on data
            if random.random() < infection_prob:
                agent.infected = True
                agent.infection_time = cabin.current_time
    
    @staticmethod
    def _surface_transmission(cabin: AircraftCabin) -> None:
        # Surface transmission more significant in 3-3-3 due to more seats
        recent_contacts = [e for e in cabin.contact_events 
                          if cabin.current_time - e['time'] < 60]
        
        surface_contacts: Dict[Tuple[float, float], List[int]] = {}
        for event in recent_contacts:
            pos = tuple(np.round(event['position'], 1))  # Group nearby positions
            if pos not in surface_contacts:
                surface_contacts[pos] = []
            surface_contacts[pos].append(event['agent_id'])
        
        for pos, agent_ids in surface_contacts.items():
            has_infected = any(cabin.get_agent_by_id(agent_id).infected 
                             for agent_id in agent_ids)
            
            if has_infected:
                for agent_id in agent_ids:
                    agent = cabin.get_agent_by_id(agent_id)
                    if not agent.infected and random.random() < 0.12:  # Slightly reduced for more seats
                        agent.infected = True
                        agent.infection_time = cabin.current_time
    