# Dataset reference

This section gives a high-level reference for the datasets supported by `prejectory`. Each page summarizes what a dataset is, the kind of traffic it captures, the expected structure on disk, and where to find the original paper or dataset source.

## At a glance

The supported datasets span several common trajectory-prediction settings:

- Highway traffic, such as [`a43`](a43.md), [`ad4che`](ad4che.md), [`exid`](exid.md), [`highd`](highd.md), [`i80`](i80.md), and [`us101`](us101.md)
- Urban intersections and mixed traffic, such as [`ind`](ind.md), [`interaction`](interaction.md), [`sind`](sind.md), and [`unid`](unid.md)
- Roundabout interaction, such as [`opendd`](opendd.md) and [`round`](round.md)
- Autonomous-driving urban datasets, such as [`argoverse1`](argoverse1.md), [`argoverse2`](argoverse2.md), [`nuscenes`](nuscenes.md), [`vod`](vod.md), and [`waymo`](waymo.md)
- Pedestrian forecasting datasets, such as [`eth`](eth.md), [`hotel`](hotel.md), [`univ`](univ.md), [`zara1`](zara1.md), and [`zara2`](zara2.md)

## How to use this section

- Start with the dataset page if you want a quick sense of whether a benchmark fits your use case.
- Use the default processing profile on each page to see the configured horizon, effective horizon,
  and known source-unit bounds before windowing.
- Check the expected structure before downloading or arranging files locally.
- Use the reference links on each page when you need the original paper, official dataset page, or benchmark source.

## Other information

In addition to this reference, use `prejectory inspect <dataset>` and
`prejectory split-support <dataset>` to get current information from the code.
