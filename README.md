# G-VUE
General *Vision* Understanding Evaluation

# Tasks

**Category 1: Perceive** 

* **Task 1.1:** Depth (NYU v2)
* **Task 1.2:** 3D Reconstruction (ShapeNet)
* **Task 1.3:** 6D Pose Estimation (LineMod)


**Category 2: Ground** 

* **Task 2.1:** Segmentation (COCO)
  * image -> dense map
* **Task 2.2:** Phrase Grounding (Ref-COCO)
  * image+text -> bounding box coordinates
* **Task 2.3:** Human-Object Interaction (V-COCO)
  * image -> bounding box coordinates
* **Task 2.4:** Vision-Language Retrieval (Flickr30k)
  * image+text -> class label

**Category 3: Reason** 

* **Task 3.1:** Visual Question Answering (GQA)
* **Task 3.2:** Bongard Problem (Bongard-HOI)
* **Task 3.3:** Common Sense Reasoning (VCR)
* **Task 2.4:** Affordance Reasoning (FunkPoint)


**Category 4: Act** 

* **Task 4.1:** Navigation (PointGoal)
* **Task 4.2:** Manipulation (CLIPort)


# Setup
Run 
```
pip install -e .
```
in the top-level directory ("G-VUE") to register packages.
