# python ecosim3d.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    TextNode,
    Vec3,
)

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText



CONFIG = {
    "world": {
        "bounds": 120.0,
        "obstacle_count": 12,
        "obstacle_radius": (4.0, 10.0),
    },
    "grid": {
        "cell_size": 12.0,
    },
    "birds": {
        "count": 120,
        "max_speed": 24.0,
        "max_accel": 35.0,
        "perception": 18.0,
        "separation_radius": 8.0,
        "weights": {
            "separation": 1.4,
            "alignment": 1.0,
            "cohesion": 0.9,
            "obstacle": 1.6,
            "wander": 0.2,
            "wind": 0.35,
        },
    },
    "predators": {
        "count": 6,
        "max_speed": 32.0,
        "max_accel": 45.0,
        "perception": 40.0,
        "weights": {
            "pursuit": 1.9,
            "obstacle": 1.8,
            "wander": 0.2,
        },
        "stamina": {
            "max": 6.0,
            "drain": 1.2,
            "recover": 0.8,
            "speed_factor": 0.55,
        },
    },
    "prey": {
        "count": 40,
        "max_speed": 12.0,
        "max_accel": 20.0,
        "perception": 20.0,
        "weights": {
            "wander": 0.6,
            "cohesion": 0.3,
            "alignment": 0.25,
            "flee": 2.0,
            "obstacle": 1.2,
        },
    },
    "scavengers": {
        "count": 20,
        "max_speed": 18.0,
        "max_accel": 28.0,
        "perception": 28.0,
        "weights": {
            "follow": 1.2,
            "opportunity": 1.5,
            "avoid": 1.7,
            "obstacle": 1.2,
            "wander": 0.3,
        },
        "follow_distance": (18.0, 32.0),
        "too_close": 10.0,
    },
}


class SpatialHash3D:
    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.cells: Dict[Tuple[int, int, int], List["EntityBase"]] = {}

    def _cell_coords(self, pos: Vec3) -> Tuple[int, int, int]:
        return (
            int(math.floor(pos.x / self.cell_size)),
            int(math.floor(pos.y / self.cell_size)),
            int(math.floor(pos.z / self.cell_size)),
        )

    def rebuild(self, entities: List["EntityBase"]) -> None:
        self.cells.clear()
        for ent in entities:
            key = self._cell_coords(ent.pos)
            self.cells.setdefault(key, []).append(ent)

    def query(self, pos: Vec3, radius: float) -> List["EntityBase"]:
        cr = int(math.ceil(radius / self.cell_size))
        cx, cy, cz = self._cell_coords(pos)
        out: List["EntityBase"] = []
        for dx in range(-cr, cr + 1):
            for dy in range(-cr, cr + 1):
                for dz in range(-cr, cr + 1):
                    out.extend(self.cells.get((cx + dx, cy + dy, cz + dz), []))
        return out


@dataclass
class EntityBase:
    world: "World"
    node: NodePath
    pos: Vec3
    vel: Vec3
    max_speed: float
    max_accel: float
    species: str
    roll: float = 0.0

    def update(self, dt: float) -> None:
        pass

    def apply_force(self, accel: Vec3, dt: float) -> None:
        if accel.length_squared() > self.max_accel ** 2:
            accel.normalize()
            accel *= self.max_accel
        self.vel += accel * dt
        speed = self.vel.length()
        if speed > self.max_speed:
            self.vel.normalize()
            self.vel *= self.max_speed
        if speed < 0.001:
            self.vel = Vec3(0, 0, 0)
        self.pos += self.vel * dt
        self.world.wrap_position(self)
        self.node.set_pos(self.pos)
        if self.vel.length_squared() > 0.001:
            look_at = self.pos + self.vel
            self.node.look_at(look_at)
            self.node.set_r(self.roll)


class Bird(EntityBase):
    def update(self, dt: float) -> None:
        cfg = CONFIG["birds"]
        neighbors = self.world.grid.query(self.pos, cfg["perception"])
        separation = Vec3(0, 0, 0)
        alignment = Vec3(0, 0, 0)
        cohesion = Vec3(0, 0, 0)
        count = 0
        for other in neighbors:
            if other is self or other.species != "birds":
                continue
            offset = self.pos - other.pos
            dist_sq = offset.length_squared()
            if dist_sq < 0.0001:
                continue
            dist = math.sqrt(dist_sq)
            if dist < cfg["perception"]:
                alignment += other.vel
                cohesion += other.pos
                count += 1
                if dist < cfg["separation_radius"]:
                    separation += offset / max(dist, 0.001)
        accel = Vec3(0, 0, 0)
        if count > 0:
            alignment = (alignment / count)
            if alignment.length_squared() > 0.001:
                alignment.normalize()
                alignment *= self.max_speed
                accel += (alignment - self.vel) * cfg["weights"]["alignment"]
            cohesion = cohesion / count - self.pos
            if cohesion.length_squared() > 0.001:
                cohesion.normalize()
                cohesion *= self.max_speed
                accel += (cohesion - self.vel) * cfg["weights"]["cohesion"]
        if separation.length_squared() > 0.001:
            separation.normalize()
            separation *= self.max_speed
            accel += (separation - self.vel) * cfg["weights"]["separation"]
        accel += self.world.avoid_obstacles(self) * cfg["weights"]["obstacle"]
        accel += self.world.wander_force(self) * cfg["weights"]["wander"]
        accel += self.world.wind_force(self.pos) * cfg["weights"]["wind"]
        self.roll = self.world.compute_roll(self.vel, accel)
        self.apply_force(accel, dt)


class Predator(EntityBase):
    stamina: float
    target: Optional[EntityBase]
    cooldown: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stamina = CONFIG["predators"]["stamina"]["max"]
        self.target = None
        self.cooldown = 0.0

    def update(self, dt: float) -> None:
        cfg = CONFIG["predators"]
        if self.cooldown > 0:
            self.cooldown -= dt
        if self.target and random.random() < 0.004:
            self.target = None
        if not self.target or not self.world.is_entity_alive(self.target):
            self.target = self.world.find_target(self.pos, ("birds", "prey"), cfg["perception"])
        accel = Vec3(0, 0, 0)
        chasing = self.target is not None
        if chasing:
            predicted = self.world.predict_target(self.pos, self.target)
            desired = predicted - self.pos
            if desired.length_squared() > 0.001:
                desired.normalize()
                desired *= self.max_speed
                accel += (desired - self.vel) * cfg["weights"]["pursuit"]
        accel += self.world.avoid_obstacles(self) * cfg["weights"]["obstacle"]
        accel += self.world.wander_force(self) * cfg["weights"]["wander"]
        stamina_cfg = cfg["stamina"]
        if chasing:
            self.stamina = max(0.0, self.stamina - stamina_cfg["drain"] * dt)
        else:
            self.stamina = min(stamina_cfg["max"], self.stamina + stamina_cfg["recover"] * dt)
        if self.stamina <= 0.1 and chasing:
            self.cooldown = 2.0
            self.target = None
        if self.stamina < stamina_cfg["max"] * 0.4:
            self.max_speed = CONFIG["predators"]["max_speed"] * stamina_cfg["speed_factor"]
        else:
            self.max_speed = CONFIG["predators"]["max_speed"]
        self.roll = self.world.compute_roll(self.vel, accel)
        self.apply_force(accel, dt)


class Prey(EntityBase):
    def update(self, dt: float) -> None:
        cfg = CONFIG["prey"]
        neighbors = self.world.grid.query(self.pos, cfg["perception"])
        cohesion = Vec3(0, 0, 0)
        alignment = Vec3(0, 0, 0)
        count = 0
        flee = Vec3(0, 0, 0)
        for other in neighbors:
            if other is self:
                continue
            offset = self.pos - other.pos
            dist_sq = offset.length_squared()
            if dist_sq < 0.0001:
                continue
            dist = math.sqrt(dist_sq)
            if other.species == "predators" and dist < cfg["perception"]:
                flee += offset / max(dist, 0.001)
            elif other.species == "prey" and dist < cfg["perception"]:
                alignment += other.vel
                cohesion += other.pos
                count += 1
        accel = Vec3(0, 0, 0)
        if count > 0:
            alignment = alignment / count
            if alignment.length_squared() > 0.001:
                alignment.normalize()
                alignment *= self.max_speed
                accel += (alignment - self.vel) * cfg["weights"]["alignment"]
            cohesion = cohesion / count - self.pos
            if cohesion.length_squared() > 0.001:
                cohesion.normalize()
                cohesion *= self.max_speed
                accel += (cohesion - self.vel) * cfg["weights"]["cohesion"]
        if flee.length_squared() > 0.001:
            flee.normalize()
            flee *= self.max_speed
            accel += (flee - self.vel) * cfg["weights"]["flee"]
        accel += self.world.wander_force(self) * cfg["weights"]["wander"]
        accel += self.world.avoid_obstacles(self) * cfg["weights"]["obstacle"]
        self.pos.z = max(0.5, self.pos.z)
        self.vel.z = 0.0
        self.roll = 0.0
        self.apply_force(accel, dt)
        self.pos.z = 0.5
        self.node.set_pos(self.pos)


class Scavenger(EntityBase):
    state: str
    state_timer: float
    target: Optional[EntityBase]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = "follow"
        self.state_timer = 0.0
        self.target = None

    def update(self, dt: float) -> None:
        cfg = CONFIG["scavengers"]
        self.state_timer -= dt
        if self.state_timer <= 0:
            self.state_timer = random.uniform(1.5, 3.5)
            self._update_state()
        accel = Vec3(0, 0, 0)
        if self.state == "follow":
            target = self.world.find_target(self.pos, ("predators",), cfg["perception"])
            if target:
                offset = target.pos - self.pos
                dist = offset.length()
                if dist > cfg["follow_distance"][1]:
                    desired = offset
                elif dist < cfg["too_close"]:
                    desired = -offset
                else:
                    desired = offset.cross(Vec3(0, 0, 1))
                if desired.length_squared() > 0.001:
                    desired.normalize()
                    desired *= self.max_speed
                    accel += (desired - self.vel) * cfg["weights"]["follow"]
        elif self.state == "opportunity" and self.target:
            if not self.world.is_entity_alive(self.target):
                self.state = "follow"
                self.target = None
            else:
                desired = self.target.pos - self.pos
                if desired.length_squared() > 0.001:
                    desired.normalize()
                    desired *= self.max_speed
                    accel += (desired - self.vel) * cfg["weights"]["opportunity"]
        elif self.state == "avoid":
            predator = self.world.find_target(self.pos, ("predators",), cfg["perception"])
            if predator:
                away = self.pos - predator.pos
                if away.length_squared() > 0.001:
                    away.normalize()
                    away *= self.max_speed
                    accel += (away - self.vel) * cfg["weights"]["avoid"]
        accel += self.world.avoid_obstacles(self) * cfg["weights"]["obstacle"]
        accel += self.world.wander_force(self) * cfg["weights"]["wander"]
        self.roll = self.world.compute_roll(self.vel, accel)
        self.apply_force(accel, dt)

    def _update_state(self) -> None:
        predators_near = self.world.find_target(self.pos, ("predators",), CONFIG["scavengers"]["perception"])
        if predators_near and (self.pos - predators_near.pos).length() < CONFIG["scavengers"]["too_close"]:
            self.state = "avoid"
            self.target = None
            return
        if random.random() < 0.4:
            target = self.world.find_opportunity(self.pos)
            if target:
                self.state = "opportunity"
                self.target = target
                return
        self.state = "follow"
        self.target = None


@dataclass
class Obstacle:
    pos: Vec3
    radius: float
    node: NodePath


class World:
    def __init__(self, base: ShowBase):
        self.base = base
        self.bounds = CONFIG["world"]["bounds"]
        self.grid = SpatialHash3D(CONFIG["grid"]["cell_size"])
        self.entities: List[EntityBase] = []
        self.obstacles: List[Obstacle] = []
        self.paused = False
        self.selected_species = "birds"
        self.selected_param_index = 0
        self.help_visible = True
        self.param_map = {
            "birds": [
                ("cohesion", CONFIG["birds"]["weights"], "cohesion"),
                ("alignment", CONFIG["birds"]["weights"], "alignment"),
                ("separation", CONFIG["birds"]["weights"], "separation"),
            ],
            "predators": [
                ("pursuit", CONFIG["predators"]["weights"], "pursuit"),
                ("max_speed", CONFIG["predators"], "max_speed"),
                ("obstacle", CONFIG["predators"]["weights"], "obstacle"),
            ],
            "prey": [
                ("flee", CONFIG["prey"]["weights"], "flee"),
                ("cohesion", CONFIG["prey"]["weights"], "cohesion"),
                ("max_speed", CONFIG["prey"], "max_speed"),
            ],
            "scavengers": [
                ("follow", CONFIG["scavengers"]["weights"], "follow"),
                ("opportunity", CONFIG["scavengers"]["weights"], "opportunity"),
                ("max_speed", CONFIG["scavengers"], "max_speed"),
            ],
        }
        self._init_scene()
        self.reset()

    def _init_scene(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.set_color((0.6, 0.6, 0.6, 1))
        self.base.render.set_light(self.base.render.attach_new_node(ambient))
        light = DirectionalLight("directional")
        light.set_color((0.9, 0.9, 0.9, 1))
        light_np = self.base.render.attach_new_node(light)
        light_np.set_hpr(45, -60, 0)
        self.base.render.set_light(light_np)

    def reset(self) -> None:
        for ent in self.entities:
            ent.node.remove_node()
        self.entities.clear()
        for obs in self.obstacles:
            obs.node.remove_node()
        self.obstacles.clear()
        self._spawn_obstacles()
        self._spawn_species("birds", Bird, CONFIG["birds"]["count"], create_cone_model((0.2, 0.6, 1.0, 1)))
        self._spawn_species("predators", Predator, CONFIG["predators"]["count"], create_cone_model((0.9, 0.2, 0.2, 1), scale=1.6))
        self._spawn_species("prey", Prey, CONFIG["prey"]["count"], create_box_model((0.4, 0.8, 0.3, 1), scale=1.2))
        self._spawn_species("scavengers", Scavenger, CONFIG["scavengers"]["count"], create_sphere_model((0.9, 0.8, 0.2, 1), scale=1.1))
        self.grid.rebuild(self.entities)

    def _spawn_species(self, species: str, cls, count: int, model: NodePath) -> None:
        for _ in range(count):
            pos = self.random_pos()
            vel = self.random_velocity()
            node = model.copy_to(self.base.render)
            entity = cls(
                world=self,
                node=node,
                pos=pos,
                vel=vel,
                max_speed=CONFIG[species]["max_speed"],
                max_accel=CONFIG[species]["max_accel"],
                species=species,
            )
            self.entities.append(entity)

    def _spawn_obstacles(self) -> None:
        count = CONFIG["world"]["obstacle_count"]
        for _ in range(count):
            radius = random.uniform(*CONFIG["world"]["obstacle_radius"])
            pos = self.random_pos(margin=radius + 8.0)
            node = create_sphere_model((0.3, 0.3, 0.35, 1), scale=radius)
            node.reparent_to(self.base.render)
            node.set_pos(pos)
            self.obstacles.append(Obstacle(pos=pos, radius=radius, node=node))

    def random_pos(self, margin: float = 0.0) -> Vec3:
        bound = self.bounds - margin
        return Vec3(
            random.uniform(-bound, bound),
            random.uniform(-bound, bound),
            random.uniform(-bound, bound),
        )

    def random_velocity(self) -> Vec3:
        vec = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        if vec.length_squared() < 0.001:
            vec = Vec3(1, 0, 0)
        vec.normalize()
        return vec * random.uniform(2.0, 6.0)

    def wrap_position(self, ent: EntityBase) -> None:
        bound = self.bounds
        if ent.pos.x > bound:
            ent.pos.x -= 2 * bound
        elif ent.pos.x < -bound:
            ent.pos.x += 2 * bound
        if ent.pos.y > bound:
            ent.pos.y -= 2 * bound
        elif ent.pos.y < -bound:
            ent.pos.y += 2 * bound
        if ent.pos.z > bound:
            ent.pos.z -= 2 * bound
        elif ent.pos.z < -bound:
            ent.pos.z += 2 * bound

    def avoid_obstacles(self, ent: EntityBase) -> Vec3:
        steer = Vec3(0, 0, 0)
        for obs in self.obstacles:
            offset = ent.pos - obs.pos
            dist_sq = offset.length_squared()
            safe = obs.radius + 6.0
            if dist_sq < safe * safe:
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.001
                steer += offset / max(dist, 0.001)
        if steer.length_squared() > 0.001:
            steer.normalize()
            steer *= ent.max_speed
            return steer - ent.vel
        return steer

    def wander_force(self, ent: EntityBase) -> Vec3:
        jitter = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        if jitter.length_squared() > 0.001:
            jitter.normalize()
        return jitter * ent.max_speed * 0.3

    def wind_force(self, pos: Vec3) -> Vec3:
        scale = 0.05
        return Vec3(
            math.sin(pos.y * scale) * 0.5,
            math.cos(pos.x * scale) * 0.3,
            math.sin((pos.x + pos.y) * scale) * 0.2,
        )

    def compute_roll(self, vel: Vec3, accel: Vec3) -> float:
        if vel.length_squared() < 0.01 or accel.length_squared() < 0.01:
            return 0.0
        forward = vel.normalized()
        right = forward.cross(Vec3(0, 0, 1))
        if right.length_squared() < 0.01:
            return 0.0
        right.normalize()
        lateral = right.dot(accel)
        return max(-35.0, min(35.0, -lateral * 0.5))

    def predict_target(self, pos: Vec3, target: EntityBase) -> Vec3:
        to_target = target.pos - pos
        distance = to_target.length()
        if distance < 0.001:
            return target.pos
        lead = distance / max(target.max_speed, 0.1)
        return target.pos + target.vel * lead

    def is_entity_alive(self, ent: EntityBase) -> bool:
        return ent in self.entities

    def find_target(self, pos: Vec3, species: Tuple[str, ...], radius: float) -> Optional[EntityBase]:
        nearest = None
        best_sq = radius * radius
        for ent in self.grid.query(pos, radius):
            if ent.species not in species:
                continue
            dist_sq = (ent.pos - pos).length_squared()
            if dist_sq < best_sq:
                best_sq = dist_sq
                nearest = ent
        return nearest

    def find_opportunity(self, pos: Vec3) -> Optional[EntityBase]:
        candidates = []
        for ent in self.grid.query(pos, CONFIG["scavengers"]["perception"]):
            if ent.species in ("birds", "prey"):
                if ent.vel.length() < ent.max_speed * 0.5:
                    candidates.append(ent)
        if not candidates:
            return None
        return min(candidates, key=lambda e: (e.pos - pos).length_squared())

    def update(self, dt: float) -> None:
        if self.paused:
            return
        self.grid.rebuild(self.entities)
        for ent in list(self.entities):
            ent.update(dt)

    def spawn_entity(self, species: str, pos: Vec3) -> None:
        model_map = {
            "birds": create_cone_model((0.2, 0.6, 1.0, 1)),
            "predators": create_cone_model((0.9, 0.2, 0.2, 1), scale=1.6),
            "prey": create_box_model((0.4, 0.8, 0.3, 1), scale=1.2),
            "scavengers": create_sphere_model((0.9, 0.8, 0.2, 1), scale=1.1),
        }
        node = model_map[species].copy_to(self.base.render)
        vel = self.random_velocity()
        cls = {
            "birds": Bird,
            "predators": Predator,
            "prey": Prey,
            "scavengers": Scavenger,
        }[species]
        entity = cls(
            world=self,
            node=node,
            pos=Vec3(pos),
            vel=vel,
            max_speed=CONFIG[species]["max_speed"],
            max_accel=CONFIG[species]["max_accel"],
            species=species,
        )
        self.entities.append(entity)

    def remove_nearest(self, pos: Vec3, radius: float = 20.0) -> None:
        nearest = None
        best_sq = radius * radius
        for ent in self.grid.query(pos, radius):
            dist_sq = (ent.pos - pos).length_squared()
            if dist_sq < best_sq:
                best_sq = dist_sq
                nearest = ent
        if nearest:
            nearest.node.remove_node()
            self.entities.remove(nearest)

    def adjust_param(self, delta: float) -> None:
        entries = self.param_map[self.selected_species]
        name, target, key = entries[self.selected_param_index]
        target[key] = max(0.05, target[key] + delta)


class CameraController:
    def __init__(self, base: ShowBase):
        self.base = base
        self.speed = 30.0
        self.fast_speed = 70.0
        self.mouse_sensitivity = 0.2
        self.dragging = False
        self.last_mouse = None
        self.move = {"forward": 0, "right": 0, "up": 0}

    def update(self, dt: float) -> None:
        if self.base.mouseWatcherNode.has_mouse() and self.dragging:
            mouse = self.base.mouseWatcherNode.get_mouse()
            if self.last_mouse:
                dx = mouse.x - self.last_mouse.x
                dy = mouse.y - self.last_mouse.y
                h = self.base.camera.get_h() - dx * 100 * self.mouse_sensitivity
                p = self.base.camera.get_p() + dy * 100 * self.mouse_sensitivity
                p = max(-89.0, min(89.0, p))
                self.base.camera.set_hpr(h, p, 0)
            self.last_mouse = mouse
        else:
            self.last_mouse = None
        speed = self.fast_speed if self.base.mouseWatcherNode.is_button_down("shift") else self.speed
        direction = Vec3(
            self.move["right"],
            self.move["forward"],
            self.move["up"],
        )
        if direction.length_squared() > 0:
            direction.normalize()
            direction *= speed * dt
            self.base.camera.set_pos(self.base.camera, direction)


class EcoSimApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.disable_mouse()
        self.camera.set_pos(0, -180, 40)
        self.camera.set_hpr(0, -10, 0)
        self.world = World(self)
        self.camera_controller = CameraController(self)
        self._build_ui()
        self._bind_controls()
        self.task_mgr.add(self._update, "update")

    def _build_ui(self) -> None:
        self._help_message = (
            "WASD: move  Q/E: up/down  Shift: fast\n"
            "Right mouse: look  Left click: spawn  Shift+Right: remove\n"
            "1-4: select species  Arrows: adjust params\n"
            "Space: pause  R: reset  H: toggle help"
        )
        self.hud_text = OnscreenText(
            text="",
            pos=(-1.3, 0.92),
            align=TextNode.ALeft,
            scale=0.05,
            fg=(1, 1, 1, 1),
            may_change=True,
        )
        self.help_text = OnscreenText(
            text=self._help_message,
            pos=(-1.3, 0.75),
            align=TextNode.ALeft,
            scale=0.045,
            fg=(0.8, 0.9, 1, 1),
            may_change=True,
        )

    def _bind_controls(self) -> None:
        self.accept("w", self._set_move, ["forward", 1])
        self.accept("w-up", self._set_move, ["forward", 0])
        self.accept("s", self._set_move, ["forward", -1])
        self.accept("s-up", self._set_move, ["forward", 0])
        self.accept("a", self._set_move, ["right", -1])
        self.accept("a-up", self._set_move, ["right", 0])
        self.accept("d", self._set_move, ["right", 1])
        self.accept("d-up", self._set_move, ["right", 0])
        self.accept("q", self._set_move, ["up", -1])
        self.accept("q-up", self._set_move, ["up", 0])
        self.accept("e", self._set_move, ["up", 1])
        self.accept("e-up", self._set_move, ["up", 0])
        self.accept("mouse3", self._start_drag)
        self.accept("mouse3-up", self._stop_drag)
        self.accept("mouse1", self._spawn_selected)
        self.accept("shift-mouse3", self._remove_nearest)
        self.accept("1", self._select_species, ["birds"])
        self.accept("2", self._select_species, ["predators"])
        self.accept("3", self._select_species, ["prey"])
        self.accept("4", self._select_species, ["scavengers"])
        self.accept("arrow_up", self._adjust_param, [0.1])
        self.accept("arrow_down", self._adjust_param, [-0.1])
        self.accept("arrow_left", self._cycle_param, [-1])
        self.accept("arrow_right", self._cycle_param, [1])
        self.accept("space", self._toggle_pause)
        self.accept("r", self._reset_world)
        self.accept("h", self._toggle_help)

    def _set_move(self, axis: str, value: int) -> None:
        self.camera_controller.move[axis] = value

    def _start_drag(self) -> None:
        self.camera_controller.dragging = True

    def _stop_drag(self) -> None:
        self.camera_controller.dragging = False

    def _toggle_pause(self) -> None:
        self.world.paused = not self.world.paused

    def _reset_world(self) -> None:
        self.world.reset()

    def _toggle_help(self) -> None:
        self.world.help_visible = not self.world.help_visible
        self.help_text.set_text("" if not self.world.help_visible else self._help_message)

    def _spawn_selected(self) -> None:
        if not self.mouseWatcherNode.has_mouse():
            return
        cam_forward = self.camera.get_quat().get_forward()
        pos = self.camera.get_pos(self.render) + cam_forward * 20
        self.world.spawn_entity(self.world.selected_species, pos)

    def _remove_nearest(self) -> None:
        self.world.remove_nearest(self.camera.get_pos(self.render))

    def _select_species(self, species: str) -> None:
        self.world.selected_species = species
        self.world.selected_param_index = 0

    def _adjust_param(self, delta: float) -> None:
        self.world.adjust_param(delta)

    def _cycle_param(self, step: int) -> None:
        entries = self.world.param_map[self.world.selected_species]
        self.world.selected_param_index = (self.world.selected_param_index + step) % len(entries)

    def _update(self, task: Task) -> int:
        dt = globalClock.get_dt()
        self.world.update(dt)
        self.camera_controller.update(dt)
        self._update_hud()
        return Task.cont

    def _update_hud(self) -> None:
        fps = globalClock.get_average_frame_rate()
        counts = {
            "birds": 0,
            "predators": 0,
            "prey": 0,
            "scavengers": 0,
        }
        for ent in self.world.entities:
            counts[ent.species] += 1
        entries = self.world.param_map[self.world.selected_species]
        param_name, target, key = entries[self.world.selected_param_index]
        hud = (
            f"FPS: {fps:5.1f}\n"
            f"Birds: {counts['birds']}  Predators: {counts['predators']}  Prey: {counts['prey']}  Scavengers: {counts['scavengers']}\n"
            f"Selected: {self.world.selected_species}  Param: {param_name} = {target[key]:.2f}"
        )
        self.hud_text.set_text(hud)


def create_cone_model(color: Tuple[float, float, float, float], scale: float = 1.0) -> NodePath:
    segments = 8
    height = 2.0 * scale
    radius = 0.6 * scale
    vformat = GeomVertexFormat.get_v3n3()
    vdata = GeomVertexData("cone", vformat, Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    tris = GeomTriangles(Geom.UH_static)
    tip = Vec3(0, height, 0)
    for i in range(segments):
        angle = (i / segments) * math.tau
        next_angle = ((i + 1) / segments) * math.tau
        p1 = Vec3(math.cos(angle) * radius, 0, math.sin(angle) * radius)
        p2 = Vec3(math.cos(next_angle) * radius, 0, math.sin(next_angle) * radius)
        base_index = i * 3
        vertex.add_data3(tip)
        vertex.add_data3(p1)
        vertex.add_data3(p2)
        n = (p1 - tip).cross(p2 - tip)
        if n.length_squared() < 0.001:
            n = Vec3(0, 1, 0)
        n.normalize()
        normal.add_data3(n)
        normal.add_data3(n)
        normal.add_data3(n)
        tris.add_vertices(base_index, base_index + 1, base_index + 2)
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode("cone")
    node.add_geom(geom)
    np = NodePath(node)
    np.set_color(color)
    np.set_two_sided(True)
    return np


def create_box_model(color: Tuple[float, float, float, float], scale: float = 1.0) -> NodePath:
    vformat = GeomVertexFormat.get_v3n3()
    vdata = GeomVertexData("box", vformat, Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    tris = GeomTriangles(Geom.UH_static)
    size = 0.9 * scale
    faces = [
        (Vec3(1, 0, 0), [Vec3(size, -size, -size), Vec3(size, size, -size), Vec3(size, size, size), Vec3(size, -size, size)]),
        (Vec3(-1, 0, 0), [Vec3(-size, -size, -size), Vec3(-size, -size, size), Vec3(-size, size, size), Vec3(-size, size, -size)]),
        (Vec3(0, 1, 0), [Vec3(-size, size, -size), Vec3(-size, size, size), Vec3(size, size, size), Vec3(size, size, -size)]),
        (Vec3(0, -1, 0), [Vec3(-size, -size, -size), Vec3(size, -size, -size), Vec3(size, -size, size), Vec3(-size, -size, size)]),
        (Vec3(0, 0, 1), [Vec3(-size, -size, size), Vec3(size, -size, size), Vec3(size, size, size), Vec3(-size, size, size)]),
        (Vec3(0, 0, -1), [Vec3(-size, -size, -size), Vec3(-size, size, -size), Vec3(size, size, -size), Vec3(size, -size, -size)]),
    ]
    idx = 0
    for n, verts in faces:
        vertex.add_data3(verts[0])
        vertex.add_data3(verts[1])
        vertex.add_data3(verts[2])
        vertex.add_data3(verts[3])
        for _ in range(4):
            normal.add_data3(n)
        tris.add_vertices(idx, idx + 1, idx + 2)
        tris.add_vertices(idx, idx + 2, idx + 3)
        idx += 4
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode("box")
    node.add_geom(geom)
    np = NodePath(node)
    np.set_color(color)
    return np


def create_sphere_model(color: Tuple[float, float, float, float], scale: float = 1.0) -> NodePath:
    vformat = GeomVertexFormat.get_v3n3()
    vdata = GeomVertexData("sphere", vformat, Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    tris = GeomTriangles(Geom.UH_static)
    points = [
        Vec3(0, 0, scale),
        Vec3(0, 0, -scale),
        Vec3(scale, 0, 0),
        Vec3(-scale, 0, 0),
        Vec3(0, scale, 0),
        Vec3(0, -scale, 0),
    ]
    faces = [
        (0, 4, 2),
        (0, 2, 5),
        (0, 5, 3),
        (0, 3, 4),
        (1, 2, 4),
        (1, 5, 2),
        (1, 3, 5),
        (1, 4, 3),
    ]
    idx = 0
    for a, b, c in faces:
        pa, pb, pc = points[a], points[b], points[c]
        vertex.add_data3(pa)
        vertex.add_data3(pb)
        vertex.add_data3(pc)
        for p in (pa, pb, pc):
            n = Vec3(p)
            if n.length_squared() > 0.001:
                n.normalize()
            normal.add_data3(n)
        tris.add_vertices(idx, idx + 1, idx + 2)
        idx += 3
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode("sphere")
    node.add_geom(geom)
    np = NodePath(node)
    np.set_color(color)
    np.set_two_sided(True)
    return np


if __name__ == "__main__":
    app = EcoSimApp()
    app.run()
