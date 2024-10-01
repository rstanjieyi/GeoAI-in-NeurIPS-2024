# <p align=center>`GeoAI in NeurIPS 2024`</p>

:star2:**A collection of papers related to Geo-spatial Information Science in NeurIPS 2024.**

## 📢 Latest Updates
:fire::fire::fire: Last Updated on 2024.09.28 :fire::fire::fire:

- **2024.9.28**: Update 14 papers.


## :memo: NeurIPS 2024 Accepted Paper List


#### FUSU: A Multi-temporal-source Land Use Change Segmentation Dataset for Fine-grained Urban Semantic Understanding

> Shuai Yuan, Guancong Lin, Lixian Zhang, Runmin Dong, Jinxiao Zhang, Shuang Chen, Juepeng Zheng, Jie Wang, Haohuan Fu

* Paper: https://arxiv.org/pdf/2405.19055
* Dataset and code: https://github.com/yuanshuai0914/FUSU
* <details>
    <summary>Abstract (Click to expand):</summary>
  Fine urban change segmentation using multi-temporal remote sensing images is essential for understanding human-environment interactions in urban areas. Although there have been advances in high-quality land cover datasets that reveal the physical features of urban landscapes, the lack of fine-grained land use datasets hinders a deeper understanding of how human activities are distributed across landscapes and the impact of these activities on the environment, thus constraining proper technique development. To address this, we introduce FUSU, the first fine-grained land use change segmentation dataset for Fine-grained Urban Semantic Understanding. FUSU features the most detailed land use classification system to date, with 17 classes and 30 billion pixels of annotations. It includes bi-temporal high-resolution satellite images with 0.2-0.5 m ground sample distance and monthly optical and radar satellite time series, covering 847 km^2 across five urban areas in the southern and northern of China with different geographical features. The fine-grained land use pixel-wise annotations and high spatial-temporal resolution data provide a robust foundation for developing proper deep learning models to provide contextual insights on human activities and urbanization. To fully leverage FUSU, we propose a unified time-series architecture for both change detection and segmentation. We benchmark FUSU on various methods for several tasks. Dataset and code are available at: https://github.com/yuanshuai0914/FUSU.
  </details>


#### SynRS3D: A Synthetic Dataset for Global 3D Semantic Understanding from Monocular Remote Sensing Imagery 
| :pushpin: Spotlight |

> Jian Song, Hongruixuan Chen, Weihao Xuan, Junshi Xia, Naoto YOKOYA

* Paper: https://arxiv.org/abs/2406.18151v1
* Dataset and code: https://github.com/JTRNEO/SynRS3D
* <details>
    <summary>Abstract (Click to expand):</summary>
    Global semantic 3D understanding from single-view high-resolution remote sensing (RS) imagery is crucial for Earth Observation (EO). However, this task faces significant challenges due to the high costs of annotations and data collection, as well as geographically restricted data availability. To address these challenges, synthetic data offer a promising solution by being easily accessible and thus enabling the provision of large and diverse datasets. We develop a specialized synthetic data generation pipeline for EO and introduce \textit{SynRS3D}, the largest synthetic RS 3D dataset. SynRS3D comprises 69,667 high-resolution optical images that cover six different city styles worldwide and feature eight land cover types, precise height information, and building change masks. To further enhance its utility, we develop a novel multi-task unsupervised domain adaptation (UDA) method, \textit{RS3DAda}, coupled with our synthetic dataset, which facilitates the RS-specific transition from synthetic to real scenarios for land cover mapping and height estimation tasks, ultimately enabling global monocular 3D semantic understanding based on synthetic data. Extensive experiments on various real-world datasets demonstrate the adaptability and effectiveness of our synthetic dataset and proposed RS3DAda method. SynRS3D and related codes will be available.
  </details>


#### Segment Any Change 

> Zhuo Zheng, Yanfei Zhong, Liangpei Zhang, Stefano Ermon

* Paper: https://arxiv.org/abs/2402.01188
* Code: https://github.com/Z-Zheng/pytorch-change-models
* <details>
    <summary>Abstract (Click to expand):</summary>
    Visual foundation models have achieved remarkable results in zero-shot image classification and segmentation, but zero-shot change detection remains an open problem. In this paper, we propose the segment any change models (AnyChange), a new type of change detection model that supports zero-shot prediction and generalization on unseen change types and data distributions.AnyChange is built on the segment anything model (SAM) via our training-free adaptation method, bitemporal latent matching.By revealing and exploiting intra-image and inter-image semantic similarities in SAM's latent space, bitemporal latent matching endows SAM with zero-shot change detection capabilities in a training-free way. We also propose a point query mechanism to enable AnyChange's zero-shot object-centric change detection capability.We perform extensive experiments to confirm the effectiveness of AnyChange for zero-shot change detection.AnyChange sets a new record on the SECOND benchmark for unsupervised change detection, exceeding the previous SOTA by up to 4.4\% F1 score, and achieving comparable accuracy with negligible manual annotations (1 pixel per image) for supervised change detection.
  </details>


#### SSDiff: Spatial-spectral Integrated Diffusion Model for Remote Sensing Pansharpening 

> Yu Zhong, Xiao Wu, Liang-Jian Deng, ZiHan Cao, Hong-Xia Dou

* Paper: https://arxiv.org/abs/2404.11537
* Code: Null
* <details>
    <summary>Abstract (Click to expand):</summary>
    Pansharpening is a significant image fusion technique that merges the spatial content and spectral characteristics of remote sensing images to generate high-resolution multispectral images. Recently, denoising diffusion probabilistic models have been gradually applied to visual tasks, enhancing controllable image generation through low-rank adaptation (LoRA). In this paper, we introduce a spatial-spectral integrated diffusion model for the remote sensing pansharpening task, called SSDiff, which considers the pansharpening process as the fusion process of spatial and spectral components from the perspective of subspace decomposition. Specifically, SSDiff utilizes spatial and spectral branches to learn spatial details and spectral features separately, then employs a designed alternating projection fusion module (APFM) to accomplish the fusion. Furthermore, we propose a frequency modulation inter-branch module (FMIM) to modulate the frequency distribution between branches. The two components of SSDiff can perform favorably against the APFM when utilizing a LoRA-like branch-wise alternative fine-tuning method. It refines SSDiff to capture component-discriminating features more sufficiently. Finally, extensive experiments on four commonly used datasets, i.e., WorldView-3, WorldView-2, GaoFen-2, and QuickBird, demonstrate the superiority of SSDiff both visually and quantitatively. The code will be made open source after possible acceptance.
  </details>


#### TorchSpatial: A Location Encoding Framework and Benchmark for Spatial Representation Learning

> Nemin Wu, Qian Cao, Zhangyu Wang, Zeping Liu, Yanlin Qi, Jielu Zhang, Joshua Ni, X. Yao, Hongxu Ma, Lan Mu, Stefano Ermon, Tanuja Ganu, Akshay Nambi, Ni Lao, Gengchen Mai

* Paper: https://arxiv.org/abs/2406.15658
* Code: https://github.com/seai-lab/TorchSpatial
* <details>
    <summary>Abstract (Click to expand):</summary>
    Spatial representation learning (SRL) aims at learning general-purpose neural network representations from various types of spatial data (e.g., points, polylines, polygons, networks, images, etc.) in their native formats. Learning good spatial representations is a fundamental problem for various downstream applications such as species distribution modeling, weather forecasting, trajectory generation, geographic question answering, etc. Even though SRL has become the foundation of almost all geospatial artificial intelligence (GeoAI) research, we have not yet seen significant efforts to develop an extensive deep learning framework and benchmark to support SRL model development and evaluation. To fill this gap, we propose TorchSpatial, a learning framework and benchmark· for location (point) encoding, which is one of the most fundamental data types of spatial representation learning. TorchSpatial contains three key components: 1) a unified location encoding framework that consolidates 15 commonly recognized location encoders, ensuring scalability and reproducibility of the implementations; 2) the LocBench benchmark tasks encompassing 7 geo-aware image classification and 4 geo-aware image regression datasets; 3) a comprehensive suite of evaluation metrics to quantify geo-aware models’ overall performance as well as their geographic bias, with a novel Geo-Bias Score metric. Finally, we provide a detailed analysis and insights into the model performance and geographic bias of different location encoders. We believe TorchSpatial will foster future advancement of spatial representation learning and spatial fairness in GeoAI research. The TorchSpatial model framework, LocBench, and Geo-Bias Score evaluation framework are available at https://github.com/seai-lab/TorchSpatial.
  </details>


#### OpenSatMap: A Fine-grained High-resolution Satellite Dataset for Large-scale Map Construction

> Hongbo Zhao, Lue Fan, Yuntao Chen, Haochen Wang, yuran Yang, Xiaojuan Jin, YIXIN ZHANG, GAOFENG MENG, ZHAO-XIANG ZHANG

* Project: https://opensatmap.github.io/
* <details>
    <summary>Abstract (Click to expand):</summary>
    In this paper, we propose OpenSatMap, a fine-grained, high-resolution satellite dataset for large-scale map construction. Map construction is one of the foundations of the transportation industry, such as navigation and autonomous driving. Extracting road structures from satellite images is an efficient way to construct large-scale maps. However, existing satellite datasets provide only coarse semantic-level labels with a relatively low resolution (up to level 19), impeding the advancement of this field. In contrast, the proposed OpenSatMap (1) has fine-grained instance-level annotations; (2) consists of high-resolution images (level 20); (3) is currently the largest one of its kind; (4) collects data with high diversity. Moreover, OpenSatMap covers and aligns with the popular nuScenes dataset and Argoverse 2 dataset to potentially advance autonomous driving technologies. By publishing and maintaining the dataset, we provide a high-quality benchmark for satellite-based map construction and downstream tasks like autonomous driving.
  </details>


#### AllClear: A Comprehensive Dataset and Benchmark for Cloud Removal in Satellite Imagery

> Hangyu Zhou, Chia-Hsiang Kao, Cheng Perng Phoo, Utkarsh Mall, Bharath Hariharan, Kavita Bala

* Project: https://allclear.cs.cornell.edu/
* <details>
    <summary>Abstract (Click to expand):</summary>
    Clouds in satellite imagery pose a significant challenge for downstream applications.A major challenge in current cloud removal research is the absence of a comprehensive benchmark and a sufficiently large and diverse training dataset.To address this problem, we introduce the largest public dataset -- *AllClear* for cloud removal, featuring 23,742 globally distributed regions of interest (ROIs) with diverse land-use patterns, comprising 4 million images in total. Each ROI includes complete temporal captures from the year 2022, with (1) multi-spectral optical imagery from Sentinel-2 and Landsat 8/9, (2) synthetic aperture radar (SAR) imagery from Sentinel-1, and (3) auxiliary remote sensing products such as cloud masks and land cover maps.We validate the effectiveness of our dataset by benchmarking performance, demonstrating the scaling law - the PSNR rises from 28.47 to 33.87 with 30× more data, and conducting ablation studies on the temporal length and the importance of individual modalities. This dataset aims to provide comprehensive coverage of the Earth's surface and promote better cloud removal results.
  </details>

#### Kuro Siwo: 33 billion $m^2$ under the water. A global multi-temporal satellite dataset for rapid flood mapping

> Nikolaos Bountos, Maria Sdraka, Angelos Zavras, Andreas Karavias, Ilektra Karasante, Themistocles Herekakis, Angeliki Thanasou, Dimitrios Michail, Ioannis Papoutsis

* Paper: https://arxiv.org/abs/2311.12056
* Code: https://github.com/Orion-AI-Lab/KuroSiwo
* <details>
    <summary>Abstract (Click to expand):</summary>
    Global floods, exacerbated by climate change, pose severe threats to human life,infrastructure, and the environment. Recent catastrophic events in Pakistan and NewZealand underscore the urgent need for precise flood mapping to guide restorationefforts, understand vulnerabilities, and prepare for future occurrences. WhileSynthetic Aperture Radar (SAR) remote sensing offers day-and-night, all-weatherimaging capabilities, its application in deep learning for flood segmentation islimited by the lack of large annotated datasets. To address this, we introduceKuro Siwo, a manually annotated multi-temporal dataset, spanning 43 flood eventsglobally. Our dataset maps more than 338 billion $m^2$ of land, with 33 billiondesignated as either flooded areas or permanent water bodies. Kuro Siwo includesa highly processed product optimized for flood mapping based on SAR GroundRange Detected, and a primal SAR Single Look Complex product with minimalpreprocessing, designed to promote research on the exploitation of both the phaseand amplitude information and to offer maximum flexibility for downstream taskpreprocessing. To leverage advances in large scale self-supervised pretrainingmethods for remote sensing data, we augment Kuro Siwo with a large unlabeled setof SAR samples. Finally, we provide an extensive benchmark, namely BlackBench,offering strong baselines for a diverse set of flood events from Europe, America,Africa, Asia and Australia.
  </details>


#### VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding

> Xiang Li, Jian Ding, Mohamed Elhoseiny

* Paper: https://arxiv.org/pdf/2406.12384
* Dataset and Code: https://github.com/lx709/VRSBench
* Project: https://vrsbench.github.io/
* <details>
    <summary>Abstract (Click to expand):</summary>
    We introduce a new benchmark designed to advance the development of general-purpose, large-scale vision-language models for remote sensing images. While several vision and language datasets in remote sensing have been proposed to pursue this goal, they often have significant limitations. Existing datasets are typically tailored to single tasks, lack detailed object information, or suffer from inadequate quality control. To address these issues, we present a versatile vision-language benchmark for remote sensing image understanding, termed VERSAL. This benchmark comprises 29,614 images, with 29,614 human-verified detailed captions, 52,472 object references, and 124,037 question-answer pairs. It facilitates the training and evaluation of vision-language models across a broad spectrum of remote sensing image understanding tasks. We further evaluated state-of-the-art models on this benchmark for three vision-language tasks: image captioning, visual grounding, and visual question answering. Our work aims to significantly contribute to the development of advanced vision-language models in the field of remote sensing.
  </details>


#### Road Network Representation Learning with the Third Law of Geography

> Haicang Zhou, Weiming Huang, Yile Chen, Tiantian He, Gao Cong, Yew Soon Ong

* Paper: https://arxiv.org/abs/2406.04038
* Code: Null

* <details>
    <summary>Abstract (Click to expand):</summary>
    Road network representation learning aims to learn compressed and effective vectorized representations for road segments that are applicable to numerous tasks. In this paper, we identify the limitations of existing methods, particularly their overemphasis on the distance effect as outlined in the First Law of Geography. In response, we propose to endow road network representation with the principles of the recent Third Law of Geography. To this end, we propose a novel graph contrastive learning framework that employs geographic configuration-aware graph augmentation and spectral negative sampling, ensuring that road segments with similar geographic configurations yield similar representations, and vice versa, aligning with the principles stated in the Third Law. The framework further fuses the Third Law with the First Law through a dual contrastive learning objective to effectively balance the implications of both laws. We evaluate our framework on two real-world datasets across three downstream tasks. The results show that the integration of the Third Law significantly improves the performance of road segment representations in downstream tasks.
  </details>

#### SolarCube: An Integrative Benchmark Dataset Harnessing Satellite and In-situ Observations for Large-scale Solar Energy Forecasting

> Ruohan Li, Yiqun Xie, Xiaowei Jia, Dongdong Wang, Yanhua Li, Yingxue Zhang, Zhihao Wang, Zhili Li

* Paper: Null
* Dataset and models: https://doi.org/10.5281/zenodo.11498739
* Code: https://github.com/Ruohan-Li/SolarCube

* <details>
    <summary>Abstract (Click to expand):</summary>
    Solar power is a critical source of renewable energy, offering significant potential to lower greenhouse gas emissions and mitigate climate change. However, the cloud induced-variability of solar radiation reaching the earth’s surface presents a challenge for integrating solar power into the grid (e.g., storage and backup management). The new generation of geostationary satellites such as GOES-16 has become an important data source for solar radiation forecasting at a large scale and high temporal frequency. However, there is no machine-learning-ready dataset that has integrated geostationary satellite data with fine-grained solar radiation information to support forecasting model development and benchmarking at a large geographic scale. We present SolarCube, a new ML-ready benchmark dataset for solar radiation forecasting. SolarCube covers 19 study areas distributed over multiple continents: North America, South America, Asia, and Oceania. The dataset supports short and long-term solar radiation forecasting at both point-level (i.e., specific locations of monitoring stations) and area-level, by processing and integrating data from multiple sources, including geostationary satellite images, physics-derived solar radiation, and ground station observations from different monitoring networks over the globe. We also evaluated a set of forecasting models for point- and image-based time-series data to develop performance benchmarks under different testing scenarios. The dataset and models are available at https://doi.org/10.5281/zenodo.11498739. The Python library to conveniently generate different variations of the dataset based on user needs is available at https://github.com/Ruohan-Li/SolarCube.
  </details>


#### G3: An Effective and Adaptive Framework for Worldwide Geolocalization Using Large Multi-Modality Models

> Pengyue Jia, Yiding Liu, Xiaopeng Li, Xiangyu Zhao, Yuhao Wang, Yantong Du, Xiao Han, Xuetao Wei, Shuaiqiang Wang, Dawei Yin

* Paper: https://arxiv.org/abs/2405.14702
* Code: https://anonymous.4open.science/r/G3-937C

* <details>
    <summary>Abstract (Click to expand):</summary>
    Worldwide geolocalization aims to locate the precise location at the coordinate level of photos taken anywhere on the Earth. It is very challenging due to 1) the difficulty of capturing subtle location-aware visual semantics, and 2) the heterogeneous geographical distribution of image data. As a result, existing studies have clear limitations when scaled to a worldwide context. They may easily confuse distant images with similar visual contents, or cannot adapt to various locations worldwide with different amounts of relevant data. To resolve these limitations, we propose G3, a novel framework based on Retrieval-Augmented Generation (RAG). In particular, G3 consists of three steps, i.e., Geo-alignment, Geo-diversification, and Geo-verification to optimize both retrieval and generation phases of worldwide geolocalization. During Geo-alignment, our solution jointly learns expressive multi-modal representations for images, GPS and textual descriptions, which allows us to capture location-aware semantics for retrieving nearby images for a given query. During Geo-diversification, we leverage a prompt ensembling method that is robust to inconsistent retrieval performance for different image queries. Finally, we combine both retrieved and generated GPS candidates in Geo-verification for location prediction. Experiments on two well-established datasets IM2GPS3k and YFCC4k verify the superiority of G3 compared to other state-of-the-art methods. Our code is available online https://anonymous.4open.science/r/G3-937C for reproduction.
  </details>


#### GOMAA-Geo: GOal Modality Agnostic Active Geo-localization

> Anindya Sarkar, Srikumar Sastry, Aleksis Pirinen, Chongjie Zhang, Nathan Jacobs, Yevgeniy Vorobeychik

* Paper: https://arxiv.org/abs/2406.01917
* Code: https://github.com/mvrl/GOMAA-Geo

* <details>
    <summary>Abstract (Click to expand):</summary>
    We consider the task of active geo-localization (AGL) in which an agent uses a sequence of visual cues observed during aerial navigation to find a target specified through multiple possible modalities. This could emulate a UAV involved in a search-and-rescue operation navigating through an area, observing a stream of aerial images as it goes. The AGL task is associated with two important challenges. Firstly, an agent must deal with a goal specification in one of multiple modalities (e.g., through a natural language description) while the search cues are provided in other modalities (aerial imagery). The second challenge is limited localization time (e.g., limited battery life, urgency) so that the goal must be localized as efficiently as possible, i.e. the agent must effectively leverage its sequentially observed aerial views when searching for the goal. To address these challenges, we propose GOMAA-Geo -- a goal modality agnostic active geo-localization agent -- for zero-shot generalization between different goal modalities. Our approach combines cross-modality contrastive learning to align representations across modalities with supervised foundation model pretraining and reinforcement learning to obtain highly effective navigation and localization policies. Through extensive evaluations, we show that GOMAA-Geo outperforms alternative learnable approaches and that it generalizes across datasets -- e.g., to disaster-hit areas without seeing a single disaster scenario during training -- and goal modalities -- e.g., to ground-level imagery or textual descriptions, despite only being trained with goals specified as aerial views.
  </details>


#### Precipitation Downscaling with Spatiotemporal Video Diffusion

> Prakhar Srivastava, Ruihan Yang, Gavin Kerrigan, Gideon Dresdner, Jeremy McGibbon, Christopher S, Bretherton, Stephan Mandt

* Paper: https://arxiv.org/abs/2312.06071
* Code: Null

* <details>
    <summary>Abstract (Click to expand):</summary>
    In climate science and meteorology, high-resolution local precipitation (rain and snowfall) predictions are limited by the computational costs of simulation-based methods. Statistical downscaling, or super-resolution, is a common workaround where a low-resolution prediction is improved using statistical approaches. Unlike traditional computer vision tasks, weather and climate applications require capturing the accurate conditional distribution of high-resolution given low-resolution patterns to assure reliable ensemble averages and unbiased estimates of extreme events, such as heavy rain. This work extends recent video diffusion models to precipitation super-resolution, employing a deterministic downscaler followed by a temporally-conditioned diffusion model to capture noise characteristics and high-frequency patterns. We test our approach on FV3GFS output, an established large-scale global atmosphere model, and compare it against six state-of-the-art baselines. Our analysis, capturing CRPS, MSE, precipitation distributions, and qualitative aspects using California and the Himalayas as examples, establishes our method as a new standard for data-driven precipitation downscaling.
  </details>


#### GeoLife: Spacial Plant Species Prediction Dataset

> Lukas Picek, Christophe Botella, Maximilien Servajean, César Leblanc, Rémi Palard, Theo Larcher, Benjamin Deneu, Diego Marcos, Pierre Bonnet, alexis joly

* Paper: https://arxiv.org/abs/2408.13928
* Project: https://www.kaggle.com/datasets/picekl/geoplant

* <details>
    <summary>Abstract (Click to expand):</summary>
    The difficulty of monitoring biodiversity at fine scales and over large areas limits ecological knowledge and conservation efforts. To fill this gap, Species Distribution Models (SDMs) predict species across space from spatially explicit features. Yet, they face the challenge of integrating the rich but heterogeneous data made available over the past decade, notably millions of opportunistic species observations and standardized surveys, as well as multi-modal remote sensing data.In light of that, we have designed and developed a new European-scale dataset for SDMs at high spatial resolution (10-50 meters), including more than 10,000 species (i.e., most of the European flora). The dataset comprises 5M heterogeneous Presence-Only records and 90k exhaustive Presence-Absence survey records, all accompanied by diverse environmental rasters (e.g., elevation, human footprint, and soil) that are traditionally used in SDMs. In addition, it provides Sentinel-2 RGB and NIR satellite images with 10 m resolution, a 20-year time-series of climatic variables, and satellite time-series from the Landsat program.In addition to the data, we provide an openly accessible SDM benchmark (hosted on Kaggle), which has already attracted an active community and a set of strong baselines for single predictor/modality and multimodal approaches.All resources, e.g., the dataset, pre-trained models, and baseline methods (in the form of notebooks), are available on Kaggle, allowing one to start with our dataset literally with two mouse clicks.
  </details>



#### Learning De-Biased Representations for Remote-Sensing Imagery

> Zichen Tian, Zhaozheng CHEN, QIANRU SUN

* Paper: Null
* Code: Null

* <details>
    <summary>Abstract (Click to expand):</summary>
    Remote sensing (RS) imagery, requiring specialized satellites to collect and being difficult to annotate, suffers from data scarcity and class imbalance in certain spectrums. Due to data scarcity, training any large-scale RS models from scratch is unrealistic, and the alternative is to transfer pre-trained models by fine-tuning or a more data-efficient method LoRA. Due to class imbalance, transferred models exhibit strong bias, where features of the major class dominate over those of the minor class. In this paper, we propose debLoRA---a generic training approach that works with any LoRA variants to yield debiased features.IncxLoRA for Incremental xLoRA. It is an unsupervised learning approach that can diversify minor class features based on the shared attributes with major classes, where the attributes are obtained by a simple step of clustering. To evaluate it, we conduct extensive experiments in two transfer learning scenarios in the RS domain: from natural to optical RS images, and from optical RS to multi-spectrum RS images. We perform object classification and oriented object detection tasks on the optical RS dataset DOTA and the SAR dataset FUSRS. Results show that our debLoRA consistently surpasses prior arts across these RS adaptation settings, yielding up to 3.3 and 4.7 percentage points gains on the tail classes for natural → optical RS and optical RS → multi-spectrum RS adaptations respectively, while preserving the performance on head classes, substantiating its efficacy and adaptability.
  </details>



#### MMM-RS: A Multi-modal, Multi-GSD, Multi-scene Remote Sensing Dataset and Benchmark for Text-to-Image Generation

> jialin luo, Yuanzhi Wang, Ziqi Gu, Yide Qiu, Shuaizhen Yao, Fuyun Wang, Chunyan Xu, Wenhua Zhang, Dan Wang, Zhen Cui

* Paper: Null
* Dataset: https://anonymous.4open.science/r/MMM-RS-C73A/

* <details>
    <summary>Abstract (Click to expand):</summary>
    Recently, the diffusion-based generative paradigm has achieved impressive general image generation capabilities with text prompt due to its accurate distribution modeling and stable training process. However, generating diverse remote sensing (RS) images that are tremendously different from general images in terms of scale and perspective remains a formidable challenge due to the lack of a comprehensive remote sensing image generation dataset with various modalities, ground sample distances (GSD), and scenes. In this paper, we propose a Multi-modal, Multi-GSD, Multi-scene Remote Sensing MMM-RS dataset and benchmark for text-to-image generation in diverse remote sensing scenarios. Specifically, we first collect nine publicly available RS datasets and conduct standardization for all samples. To bridge RS images to textual semantic information, we utilize a large-scale pretrained vision-language model to automatically output text prompt and perform hand-crafted rectification, resulting in information-rich text-image pairs (including multi-modal images).In particular, we design some methods to obtain the images with different GSD and various environments (e.g., low-light, foggy) in a single sample. With extensive manual screening and refining annotations, we ultimately obtain a MMM-RS dataset that comprises approximately 2.1 million text-image pairs. Extensive experimental results verify that our proposed MMM-RS dataset allows off-the-shelf diffusion models to generate diverse RS images across various modalities, scenes, weather conditions, and GSD. The dataset is available at https://anonymous.4open.science/r/MMM-RS-C73A/.
  </details>


  

#### COSMIC: Compress Satellite Image Efficiently via Diffusion Compensation

> Ziyuan Zhang, Han Qiu, Zhang Maosen, Jun Liu, Bin Chen, Tianwei Zhang, Hewu Li

* Paper: Null
* Code: Null

* <details>
    <summary>Abstract (Click to expand):</summary>
    With the rapidly increasing number of satellites in space and their enhanced capabilities, the amount of earth observation images collected by satellites is exceeding the transmission limits of satellite-to-ground links. Although existing learned image compression solutions achieve remarkable performance by using a sophisticated encoder to extract fruitful features as compression and using a decoder to reconstruct. It is still hard to directly deploy those complex encoders on current satellites' embedded GPUs with limited computing capability and power supply to compress images in orbit. In this paper, we propose COSMIC, a simple yet effective learned compression solution to transmit satellite images. We first design a lightweight encoder (i.e. reducing FLOPs by 2.5~5X) on satellite to achieve a high image compression ratio to save satellite-to-ground links. Then, for reconstructions on the ground, to deal with the feature extraction ability degradation due to simplifying encoders, we propose a diffusion-based model to compensate image details when decoding. Our insight is that satellite's earth observation photos are not just images but indeed multi-modal data with a nature of Text-to-Image pairing since they are collected with rich sensor data (e.g. coordinates, timestep, etc.) that can be used as the condition for diffusion generation. Extensive experiments show that COSMIC outperforms state-of-the-art baselines on both perceptual and distortion metrics.
  </details>



#### Progressive Exploration-Conformal Learning for Sparsely Annotated Object Detection in Aerial Images

> Zihan Lu, Chunyan Xu, wang chenxu, Xiangwei Zheng, Zhen Cui

* Paper: Null
* Code: Null

* <details>
    <summary>Abstract (Click to expand):</summary>
    The ability to detect aerial objects with limited annotation is pivotal to the development of real-world aerial intelligence systems. In this work, we focus on a demanding but practical sparsely annotated object detection (SAOD) in aerial images, which encompasses a wider variety of aerial scenes with the same number of annotated objects. Although most existing SAOD methods rely on fixed thresholding to filter pseudo-labels for enhancing detector performance, adapting to aerial objects proves challenging due to the imbalanced probabilities/confidences associated with predicted aerial objects. To address this problem, we propose a novel Progressive Exploration-Conformal Learning (PECL) framework to address the SAOD task, which can adaptively perform the selection of high-quality pseudo-labels in aerial images. Specifically, the pseudo-label exploration can be formulated as a decision-making paradigm by adopting a conformal pseudo-label explorer and a multi-clue selection evaluator. The conformal pseudo-label explorer learns an adaptive policy by maximizing the cumulative reward, which can decide how to select these high-quality candidates by leveraging their essential characteristics and inter-instance contextual information. The multi-clue selection evaluator is designed to evaluate the explorer-guided pseudo-label selections by providing an instructive feedback for policy optimization. Finally, the explored pseudo-labels can be adopted to guide the optimization of aerial object detector in a closed-looping progressive fashion. Comprehensive evaluations on two public datasets demonstrate the superiority of our PECL when compared with other state-of-the-art methods in the sparsely annotated aerial object detection task.
  </details>


#### DistrictNet: Decision-aware learning for geographical districting

> Cheikh Ahmed, Alexandre Forel, Axel Parmentier, Thibaut Vidal

* Paper: Null
* Code: Null

* <details>
    <summary>Abstract (Click to expand):</summary>
    Districting is a complex combinatorial problem that consists in partitioning a geographical area into small districts. In logistics, it is a major strategic decision determining operating costs for several years. Solving them using traditional methods is intractable even for small geographical areas and existing heuristics, while quick, often provide sub-optimal results. We present a structured learning approach to find high-quality solutions to real-world districting problems in a few minutes. It is based on integrating a combinatorial optimization layer, the capacitated minimum spanning tree problem, into a graph neural network architecture. To train this pipeline in a decision-aware fashion, we show how to construct target solutions embedded in a suitable space and learn from target solutions. Experiments show that our approach outperforms existing methods as it reduces costs by 10% on average on real-world cities.
  </details>