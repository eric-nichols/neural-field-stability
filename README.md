# neural-field-stability
A Python program to test neural field stability.

This code only works out-of-the box for the equations we were using on the NeuroSys project at INRIA in Nancy, France and remotely in Canada. However, I'm releasing the source code in case anyone wants to modify the equations for their own use.
The program is dependent on a wxPython installation. 

Running the code:
  1. On the left half of the interface, select parameters for the stationary state, firing rate and analysis units.
  2. Click the Get Root button to display the roots.
  3. Select a root to get the wave vector k.
  4. The system will be stable if the maximum value of f(k) is less than 1. Otherwise, it will be unstable and highlighted in red.
