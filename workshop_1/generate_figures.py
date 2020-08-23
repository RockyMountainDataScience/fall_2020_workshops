{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "module 'pandas' has no attribute 'tslib'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-5-2799f360d0fb>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mggplot\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m~/anaconda3/envs/rmds-f2020-ws1/lib/python3.8/site-packages/ggplot/__init__.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     17\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     18\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 19\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mgeoms\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mgeom_area\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_blank\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_boxplot\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_line\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_point\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_jitter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_histogram\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_density\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_hline\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_vline\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_bar\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_abline\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_tile\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_rect\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_bin2d\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_step\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_text\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_path\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_ribbon\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_now_its_art\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_violin\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_errorbar\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgeom_polygon\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     20\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mstats\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mstat_smooth\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mstat_density\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     21\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/rmds-f2020-ws1/lib/python3.8/site-packages/ggplot/geoms/__init__.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mgeom_abline\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mgeom_abline\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mgeom_area\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mgeom_area\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mgeom_bar\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mgeom_bar\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mgeom_bin2d\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mgeom_bin2d\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mgeom_blank\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mgeom_blank\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/rmds-f2020-ws1/lib/python3.8/site-packages/ggplot/geoms/geom_abline.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mgeom\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mgeom\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mclass\u001b[0m \u001b[0mgeom_abline\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mgeom\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m     \"\"\"\n\u001b[1;32m      5\u001b[0m     \u001b[0mLine\u001b[0m \u001b[0mspecified\u001b[0m \u001b[0mby\u001b[0m \u001b[0mslope\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mintercept\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/rmds-f2020-ws1/lib/python3.8/site-packages/ggplot/geoms/geom.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m from __future__ import (absolute_import, division, print_function,\n\u001b[1;32m      2\u001b[0m                         unicode_literals)\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mggplot\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mggplot\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0maes\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0maes\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/rmds-f2020-ws1/lib/python3.8/site-packages/ggplot/ggplot.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mwarnings\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     12\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 13\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0maes\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0maes\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     14\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mlegend\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mmake_legend\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mthemes\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtheme_gray\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/rmds-f2020-ws1/lib/python3.8/site-packages/ggplot/aes.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      9\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mpatsy\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0meval\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mEvalEnvironment\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 11\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mutils\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     12\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     13\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/rmds-f2020-ws1/lib/python3.8/site-packages/ggplot/utils.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     79\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     80\u001b[0m date_types = (\n\u001b[0;32m---> 81\u001b[0;31m     \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtslib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mTimestamp\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     82\u001b[0m     \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mDatetimeIndex\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     83\u001b[0m     \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mPeriod\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/rmds-f2020-ws1/lib/python3.8/site-packages/pandas/__init__.py\u001b[0m in \u001b[0;36m__getattr__\u001b[0;34m(name)\u001b[0m\n\u001b[1;32m    256\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0m_SparseArray\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    257\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 258\u001b[0;31m         \u001b[0;32mraise\u001b[0m \u001b[0mAttributeError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"module 'pandas' has no attribute '{name}'\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    259\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    260\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mAttributeError\u001b[0m: module 'pandas' has no attribute 'tslib'"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import ggplot"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "rmds-f2020-ws1",
   "language": "python",
   "name": "rmds-f2020-ws1"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
