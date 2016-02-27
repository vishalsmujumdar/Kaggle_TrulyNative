Hello, This is out ALDA project. In order for this project to work you will need to install a data science distribution of python named Anaconda.

Download this version from https://www.continuum.io/downloads

Go through the GUI Installer on Windows or Mac.

After the insraller finishes follow these steps:

Note: While running the commands mentioned below you may run into permission issues. To avoid those run the cmd as administrator or change user to root user in Terminal(by running 'su root' and then entering your password)

1. Open Command Prompt/Terminal and navigate to folder where Anaconda is installed.
2. Run command 'conda update conda'
3. Run command 'conda update anaconda'
4. Now that Ananconda is updated run this command to install some extra python packages that we might use later.
  'conda create -n alda cython distribute ipython-notebook ipython-qtconsole jinja2 lxml matplotlib nose numba numexpr pandas pip pygments pytables pywin32 scipy statsmodels xlrd xlwt csvkit'
5. After all the new packages get installed there will be a message on the cmd/terminal that will ask you to activate the environment 'alda' that we created above. To do this run 'cd Scripts' and then 'activate alda'

After this your job on the command line is done. Now comes the part of setting up the anaconda interpreter in the IDE. The recommended IDE is PyCharm. You can use any one you want. 

If you are done with this you can go through the code and try to run the sampler.py either through the IDE or on the command line. You need to provide three arguments to the sampler.py file.
1. Input directory
2. Output directory
3. Name of the training file (which in our case is train_v2.csv)

So your command should look something like this

C:\Anaconda3\envs\alda\python.exe "C:/Users/Vishal Mujumdar/PycharmProjects/TrulyNative/sampler.py" "C:/Users/Vishal Mujumdar/Desktop/CSC522/Project/Data" "C:/Users/Vishal Mujumdar/Desktop/Data" "train_v2.csv"

Once you run this you should see simple output. Check in the output folder you will also see the files.
