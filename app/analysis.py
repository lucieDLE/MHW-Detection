TIMESERIE_CAPTION = """
    The **video** shows weekly sea surface temperature **anomalies** (SSTA). SSTA are deviations 
    from the monthly climatology across the global ocean. The diverging colormap is centered on 
    zero: **red** indicates warmer-than-usual water, **blue** cooler-than-usual, and **white** 
    near-normal conditions. The color scale is clipped to ±5 ˚C. We can see a clear increase in 
    appearance of hot events with higher probabilities, spread, strength.

    - **Play/Pause**: use the video controls to start or stop playback.
    - Frames between real weekly timesteps are linearly interpolated for smooth motion.
    - Speed can also be adjusted.
"""

ANOMALY_CAPTION = """
    The above **worldmap** highlights where Sea Surface Temperature (SST) fluctuates most from year 
    to year (red). Clicking on a location generates several analyses: 

    - **Linear trend estimation**: An Ordinary Least Squares (OLS) regression is applied to estimate
        the long-term SST anomaly trend.
    - **Extreme event detection**: defined as SST anomalies exceeding a 95th percentile threshold
    - **Histogram of the number of extreme event** with Kernel Density Estimation (KDE): detects if 
        warm anomalies are becoming more frequent and how the distribution of SST anomalies evolves 
        over time.

    **Analysis**:
    Areas showing strong variability independently of seasonality are:
    
    - **El Niño-Southern Oscillation (ENSO):** Recurring climate pattern (3 to 7 years) warming 
        (El Niño) or cooling (La Niña) the water by ~1 to 3 C compared to normal SST. ENSO is one 
        of the most important climate phenomena on Earth as it changes temperatures, precipitation 
        around the globe.
    
    - **Gulf Stream (GS):** A strong ocean current bringing warm water from the Gulf of Mexico into 
        the Atlantic ocean. Hence this location has more variability than normal. The cold water 
        from the Labrador current goes down and the warmer than normal SSTs stick around. Anomalies
        are computed relative to the mean, a small displacement of the current can produces a larger
        deviation from the mean.
    
    - **Kuroshio Extension (KE):** Similar to the GS, the KE is a powerful current in the Pacific 
        from the East Coast of the Japan and meanders east towards North America. 
    
    - **Agulhas current (AC):** Strong ocean current bringing also warm water from the southeast 
        coast of Mozambique to South Africa and then meanders east toward Australia.
    
    - **Brazil-Malvinas (or Falkland) Confluence (BMC):** Confluence of 2 currents off the coast of 
        Argentina and Uruguay where the warm Brazil Current and the cold Falkland current converge.
"""

MHW_CAPTION = """
    This panel detects **Marine HeatWave (MHW)** events defined as periods of **≥5 consecutive days** 
    where SST exceeds the 90th-percentile climatological threshold. The map shows either the total 
    number of MHW **days** or discrete MHW **events** per year at each grid cell.

    - Click anywhere on the map to see the full time series per year at that location.
    - Use the metric selector and year slider to explore spatial patterns.


    **Analysis**: 

    Several recurring patterns emerge across the maps, intensifying from the 1980s to the 2020s:

    - **1983, 1997-1998, 2015-2016, and 2023-2024** correspond to major El Niño events and appear 
        as the brightest, most spatially extensive maps. El Niño also indirectly warms the Indian 
        and Atlantic Oceans later by reorganizing global wind patterns. 2016 combines a strong El
        Niño with the long-term warming trend, making it the most intense map in the series. 

    - **Long-term background warming:** Comparing **1987** and **2022** two quiet years without any
        ENSO phenomenon reveals a critical shift: the 2022 map is considerably brighter almost 
        everywhere. This illustrates the human impact on ocean warming, gradually raising the baseline
        temperature of the ocean. As a result, even ordinary years now produce more MHW days (2022) 
        than exceptional years did 40 ago (1983).

    - **1992**: A localized MHW signal appears in the Southern Ocean between Australia and South America. 
        The 1992 anomaly is likely linked to the **1991 Mount Pinatubo volcanic eruption**, which 
        disrupted global wind patterns and may have caused regional warming in this area.

    - **2005, 2011-2012**: The Arctic is the fastest-warming region on the planet because melting 
        white ice (reflective) is replaced with dark, heat-absorbing ocean water, which accelerates
        further melting. This makes the Arctic particularly sensitive to MHW conditions. Whether 
        these specific years were triggered by discrete events or by the long-term warming trend 
        remains unclear.
    \n

    The 1982-2025 period shows that **MHWs that were once rare, confined, and tied to El Niño** events
    are **now longer, more widespread and frequent** even in the absence of any specific climate 
    phenomenon. 

    **This highlights an ongoing and accelerating warming of the global ocean.**
"""
FORECAST_METRIC_CAPTION = """
    **Metrics**

    - **Anomaly Correlation Coefficient (ACC)**: Spatial Pearson correlation between predicted and 
        observed SSTA per timestep, averaged over the test period.

    - **Root Mean Square Error (RMSE)**: Per-pixel error between predicted and observed SSTA.

    - **Forecasting Skill Score**: Measures improvement over the persistence baseline. A score of 0
        means the model performs no better than persistence; 1 means a perfect forecast; a negative
        score means persistence wins. Blue regions indicate where persistence is more informative.

    **Overall**: The model has genuinely learned useful structure in the tropics, where slow ENSO
        driven signals provide a learnable target. However, the autoregressive rollout strategy is 
        too fragile for high-latitude regions, where errors compound fastest. A direct model (n_out = 28) 
        would likely address much of this degradation without requiring a more complex architecture.

    You can use the select option to dive into model performances and comparisons:

    - Select a metric and lead time to display the corresponding spatial map, accompanied by an analysis.

    - Select a time period and click anywhere on the map to visualize the predicted SSTA at that location.
"""


FORECAST_SKILL_CAPTION = """
    **Forecasting skill**

    - **Tropical band**: Across almost all lead times, the model captures SST dynamics most reliably 
        in the ENSO region. Skill remains positive up to t = 28 days, reflecting the slow, large-scale 
        nature of ENSO-driven anomalies.

    - **High latitudes**: The model underperforms persistence in the Arctic and Southern Ocean, 
        increasingly with time. High-latitude SST is driven by fast, chaotic processes (storms, 
        sea ice dynamics), where small initial errors are carried forward and amplified at each
        rollout step.

    - **Weddell Sea**: The persistently deep blue spot in the Southern Ocean is the worst-performing
        region across all lead times, including at t = 1. This area is called the **Weddell Sea**, 
        which is governed by sea ice melt and freeze cycles that are fundamentally different from
        open ocean dynamics.
"""

FORECAST_RMSE_CAPTION = """
    **RMSE Maps (Model and Persistence)**

    - Both the model and persistence share a very similar spatial error structure: highest RMSE along
        strong ocean currents (Gulf Stream, Kuroshio Extension), in high-latitude regions, and near
        coastlines.

    - This confirms that 1) the model has learned a level of predictability comparable to persistence 
        at short lead times and 2) the regions that are hardest to forecast are the same for both
        methods. These areas are known to have high SST variability.

    - By lead time 21, the model's RMSE deteriorates significantly in the most dynamically active 
        regions, while the tropical Pacific and Indian Ocean remain comparatively well-forecasted.
        This is a direct consequence of the rollout error accumulation in high-variability areas.
"""

FORECAST_ACC_CAPTION = """
    **ACC Maps (Model and Persistence)**

    - Both methods maintain high ACC (deep red, close to 1) across most of the global ocean at all 
        lead times. However, this is expected: ACC measures correlation, and SST anomalies change 
        slowly, making them easier to predict over short to medium horizons, and therefore easier 
        to correlate with observations.

    - The most informative regions are where the maps fade from red toward yellow/white: strong 
        currents and high-latitude areas associated with high and fast SST variability. By t=28 days,
        the model appears to achieve slightly better performance in these regions, but the difference 
        is very subtle. 

    - This is why the ACC difference maps are more informative than these absolute ACC maps alone.
"""

FORECAST_DIFF_ACC_CAPTION = """
    **ACC Difference Map**

    - Up to t = 10 days: The ACC difference is mostly neutral globally, with a clear positive band
        along the equatorial Pacific confirming that the model outperforms persistence in the ENSO 
        region. High latitudes show a mild negative signal but remain close to zero.

    - t = 14 to t = 28 days: The spatial structure degrades rapidly. The Arctic and Southern Ocean 
        turn strongly blue, with the model losing up to 0.4 ACC points relative to persistence.

    - At t = 28 days: The model adds very little value over persistence at the global scale, with 
        meaningful positive skill confined almost entirely to the tropical Pacific.
"""
