Source code to help people reproduce the graphs presented at [Data Berlin](https://www.linkedin.com/posts/business-intelligence-berlin_in-the-summertime-we-do-another-meetup-activity-7216443318962974721-4x-4/).

![Statistical power with 80k Users and 10% uplift](/assets/img.png)

The test_methods notebook is a sandbox to play around with for different scenarios.

You can tweak the following parameter set:

```
NN: Number of experiments
N: Number of users per variant
uplift_purchase_rate: the uplift in number of purchases per player
uplift_purchase_amount: the uplift in purchase value per player
```

If you want to adapt the generated dataset itself to tweak it more to your needs, you have to adapt the **generation.py** script.

