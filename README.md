## Code to inferencing with Retinaface original model converted to onnx
Currently the tested models are original Retinaface (r50) converted to 640x640, 1024x1280, 900x900 with dynamic batch sizes (from [official Retinaface](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace)).
To convert the model to onnx we used [codes from here](https://github.com/SthPhoenix/InsightFace-REST).

### Running
- Install all required packages
> pip install -r requirements.txt
- Place model in model folder
- Place test images in folder images
- Run test with:
> python face_detection --model path/to/model --gpu 0 --images /path/to/test/images/ --scales 900,900 --batching 16 
  - scales - size of image that model can receive (e.g. 640x640, 800x800, 900x900, etc.)
  - batching - denotes batch size (if not set the code will run without batching)

By default batch results will be saved in batch_resized and batch_results folders and non-batch results are saved in resized and results folders. 

### Issues
If you are using other than x86 platforms, you can encounter bbox problems. One solution can be to recompile using Cython:
  - Go to rcnn/cython and do (you have to have Cython package installed):
  > python setup.py build_ext --inplace

### TO-DO
- [ ] Writing to folder by using multiple processes
- [ ] Speed up postprocessing
- [ ] Alignment of faces
- [ ] Recognition
- [ ] Integrating faiss
- [ ] Creating API