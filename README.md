# Advanced ML course (ITMO)
The project contains solutions for some labs from the [advanced machine learning course][course] made at ITMO.
To make environment and install dependencies:
```sh
./setup.sh
```
## SVM
To run (in this example there are default values provided for `train` and `test` argument so in this case they could have been omitted):
```
python -m svm classify-images --train resources/svm/train/ --test resources/svm/test
```

[course]: https://courses.openedu.ru/courses/course-v1:ITMOUniversity+ADVML+spring_2020_ITMO_mag/courseware/f2d9d5651e64479b8aece0048a66db87/97db5f63cbc54030a471753d0cd14332/
