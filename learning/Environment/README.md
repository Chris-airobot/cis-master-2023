## Environment Logic
Two agents game, one is a prisoner, trying to escape. The other one is a helper, it can create route for the prisoner to escape. This game is displayed on a 7x7 grid, where:

- The prisoner starts in the top left corner,
- The escape door is randomly placed in the middle of the grid
- All other places except escape door and the prisoner's initial positions are death traps.
- The helper could build a brige on top of one trap each time.
- The prisoner cannot go through the trap

### Actions
- Solver action: up/down/left/right/no movements
- Create bridge that nears the solver
