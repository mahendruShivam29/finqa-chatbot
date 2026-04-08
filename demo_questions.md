# Demo Questions

These questions were validated against the current saved global retriever using the plain question text only, with no added helper context.

Each entry includes:
- the exact question
- the sample reference in `data/...`
- the sample `id`
- the expected ground-truth answer from `sample["qa"]["exe_ans"]`

## Validated Questions

1. Question: `during the 2012 year , did the equity awards in which the prescribed performance milestones were achieved exceed the equity award compensation expense for equity granted during the year?`
   Reference: `data/train.json`
   Sample ID: `ABMD/2012/page_75.pdf-1`
   Expected Answer: `yes`

2. Question: `what percentage of total cash and investments as of dec . 29 2012 was comprised of available-for-sale investments?`
   Reference: `data/train.json`
   Sample ID: `INTC/2013/page_71.pdf-4`
   Expected Answer: `0.53232`

3. Question: `what is the growth rate in net revenue in 2008?`
   Reference: `data/train.json`
   Sample ID: `ETR/2008/page_313.pdf-3`
   Expected Answer: `-0.03219`

4. Question: `what was the growth rate of the loans held-for-sale that are carried at locom from 2009 to 2010`
   Reference: `data/train.json`
   Sample ID: `C/2010/page_272.pdf-1`
   Expected Answer: `0.97656`

5. Question: `in millions , what is the total of home equity lines of credit?`
   Reference: `data/train.json`
   Sample ID: `PNC/2012/page_110.pdf-3`
   Expected Answer: `22929.0`

6. Question: `what is the percent of the labor-related deemed claim as part of the total reorganization items net in 2013`
   Reference: `data/train.json`
   Sample ID: `AAL/2015/page_74.pdf-1`
   Expected Answer: `0.65273`

7. Question: `what is the percent change in net revenue between 2007 and 2008?`
   Reference: `data/train.json`
   Sample ID: `ETR/2008/page_376.pdf-2`
   Expected Answer: `-0.00317`

8. Question: `in 2013 what percentage of total net revenues for the investing & lending segment were due to debt securities and loans?`
   Reference: `data/train.json`
   Sample ID: `GS/2014/page_47.pdf-3`
   Expected Answer: `0.27743`

9. Question: `what percentage of total reorganization items net consisted of labor-related deemed claim?`
   Reference: `data/train.json`
   Sample ID: `AAL/2014/page_89.pdf-3`
   Expected Answer: `0.65644`

10. Question: `as part of the proceeds from the clear wire transactions what was the percent of the gain recognized included in the equity investments , net on the consolidated statements of income .`
    Reference: `data/train.json`
    Sample ID: `INTC/2013/page_71.pdf-2`
    Expected Answer: `0.93404`

11. Question: `on what percent of trading days were there market gains above $ 210 million?`
    Reference: `data/train.json`
    Sample ID: `JPM/2010/page_144.pdf-2`
    Expected Answer: `0.04598`

12. Question: `in february 2016 what was the percent reduction in the board of directors authorized the repurchase to the february 2014`
    Reference: `data/train.json`
    Sample ID: `MMM/2015/page_19.pdf-2`
    Expected Answer: `-0.16667`

13. Question: `in millions between 2014 and 2013 , what was the change in net derivative liabilities under bilateral agreements?`
    Reference: `data/train.json`
    Sample ID: `GS/2014/page_134.pdf-3`
    Expected Answer: `13588.0`

14. Question: `what was the net notional amounts of purchases and sales under sfas 140 in 2003 ( us$ b ) ?`
    Reference: `data/train.json`
    Sample ID: `JPM/2003/page_100.pdf-4`
    Expected Answer: `7.0`

15. Question: `what is the percentage of consolidated communities among the total communities?`
    Reference: `data/train.json`
    Sample ID: `MAA/2018/page_19.pdf-2`
    Expected Answer: `0.99671`

16. Question: `what portion of total backlog is related to ingalls segment?`
    Reference: `data/train.json`
    Sample ID: `HII/2018/page_64.pdf-4`
    Expected Answer: `0.37399`

17. Question: `what is percentage change in total conduit asset from 2007 to 2008?`
    Reference: `data/train.json`
    Sample ID: `STT/2008/page_116.pdf-1`
    Expected Answer: `-0.16849`

18. Question: `what was the difference in operating profit margins as adjusted from 2015 to 2016?`
    Reference: `data/train.json`
    Sample ID: `MAS/2017/page_37.pdf-1`
    Expected Answer: `0.016`

19. Question: `what was the total amount lost from the bond authorization to the withdrawn?`
    Reference: `data/train.json`
    Sample ID: `TSCO/2017/page_68.pdf-3`
    Expected Answer: `13.0`

20. Question: `what was the total fees earned in 2016 for management , leasing and construction and development`
    Reference: `data/train.json`
    Sample ID: `DRE/2016/page_64.pdf-4`
    Expected Answer: `14.9`
