# Push-Net: Deep Recurrent Neual Network for Planar Pushing Objects of Unknown Physical Properties

## What is Push-Net ?
* a deep recurrent neural network that selects actions to push objects with unknown phyical properties
* unknown physical properties: center of mass, mass distribution, friction coefficients etc.
* for technical details, refer to the [paper](http://motion.comp.nus.edu.sg/wp-content/uploads/2018/06/rss18push.pdf)

## Environment and Dependencies
* Ubuntu 14.04
* Python 2.7.6
* Pytorch 0.3.1
* GPU: GTX 980M
* CUDA version: 8.0.44
* [imutils](https://github.com/jrosebr1/imutils)

## Create A Virtualenv for Push-Net
* create virtualenv dependent on system python2.7

```virtualenv -p /usr/bin/python2.7 venv```

* activate venv

```source venv/bin/activate```

* install pytorch 0.3.1

```pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl ```

```pip install torchvision ```

* download opencv2.4 source from [link](https://opencv.org/releases.html)
* install opencv from source (needed for python2.7)

```cd opencv-2.4.13.6```

```mkdir release & cd release```

```cmake -D MAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/local/ -D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python -D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python2.7/site-packages -D INSTALL_PYTHON_EXAMPLES=ON ..```

``` make -j8 ```

``` make install```

Note: if you encounter the following compilation error while building opencv

``` Error: /modules/contrib/src/rgbdodometry.cpp:65:47: fatal error: unsupported/Eigen/MatrixFunctions: No such file or directory ```

You have to find the path of unsupported/Eigen/MatrixFunctions. In my case it was inside /usr/include/eigen3/.

Then to solve the problem you have to open modules/contrib/src/rgbdodometry.cpp and add "eigen3/" to the include path at line 65.

* if you want to use rospy in virtualenv

```pip install rospkg catkin_pkg```


## Usage
* Input: an current input image mask of size 128 x 106, and goal specification (see [push_net_main.py](push_net_main.py))
* Output: the best push action on the image plane
* Example:
  
  input image : ```test.jpg```
  
  ```python push_net_main.py```
  
  result: the input image with the best action (red arrow) will be displayed
  

## License and Citation
* See [LICENSE](LICENSE.md) file for license rights and limitations (GNU)
* If you are using part of the code for your research work, kindly cite this work

``` 
J.K. Li, D. Hsu, and W.S. Lee. Push-Net: Deep planar pushing for objects with unknown physical properties. In Proc. Robotics: Science & Systems, 2018.
```
OR bibtex: 

```
@inproceedings{Li2018PushNet,
  title={Push-Net : Deep Planar Pushing for Objects with Unknown Physical Properties},
  author={Jue Kun Li and David Hsu and Wee Sun Lee},
  booktitle={Robotics: Science and System),
  year={2018}
}
```





