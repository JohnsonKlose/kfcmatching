# kfcmatching
此项目以百度地图和高德地图中肯德基门店数据为基础，建立两组数据的相似性计算方法，通过相似性计算结果找出百度和高德地图中表示同一店铺的数据，同时运用SVM和Logistic Regression的方法，构建自动化筛选匹配结果的机器学习模型。
## 数据准备
- **数据获取：**  
    通过抓包抓取百度地图和高德地图肯德基门店数据，数据包括门店名称、门店地址、所在区域、门店标签、地理坐标等内容，具体爬虫方法请移步[https://github.com/JohnsonKlose/kfcscrape](https://github.com/JohnsonKlose/kfcscrape)  
- **数据清洗：**  
    获取数据后，查看到数据中有很多是肯德基甜品站、停车场或卫生间，因此做一个简单的数据清洗，去除掉这些我们不需要的数据。同时再针对个别数据做一些调整，例如把门店名称中包含“KFC”的内容全部替换为“肯德基”等。  
    下面是做数据清洗时判断门店名称是否符合我们需要的函数，返回的是布尔值：  
    ```
    def iskfc(text):
        if "洗手间" in text:
            return False
        if "宅急送" in text:
            return False
        if "甜品站" in text:
            return False
        if "KFC" in text:
            return False
        if "中国人的肯德基" == text:
            return False
        else:
            return True
    ```  
    **Tips:** 数据存储在数据库中，完全可以通过SQL在数据库中操作完成数据清洗工作，也更加方便快捷。  
    
## 相似性计算
一条百度地图的肯德基门店数据和一条高德地图的肯德基门店数据相匹配，我们可以通过计算相似度的定量方法来判断两条数据是否是可以匹配的相同门店。相似度计算共包括空间相似性和语义相似性两部分，下面分别对空间相似性和语义相似性的计算方法做详细的描述：
- **空间相似性**  
    空间相似性描述的是两条数据坐标相距的距离远近。具体方法如下：  
    1.
        百度地图数据爬取的坐标是形如{dipointx:1322465885,dipointy:374594834}这样的数据，这其实是墨卡托投影下的平面坐标数据，为了方便进行坐标距离计算，我们把这样的坐标通过百度地图JavaScript API提供的方法转换成经纬度坐标。转换方法如下所示：  
        ```
        var projection =new BMap.MercatorProjection();
        var point = projection.pointToLngLat(new BMap.Pixel(x,y))
        ```
        详细的转换方法请移步[https://github.com/JohnsonKlose/mercatorprojection](https://github.com/JohnsonKlose/mercatorprojection)
    2.
        需要注意的是，通过上一步的平面坐标向经纬度坐标的转换，百度地图坐标数据的坐标系是bd09坐标系(百度自己对经纬度的加密算法)，而高德地图坐标数据的坐标系是gcj02坐标系(中国国家测绘局制订的地理信息系统的坐标系统)，因此还需要把这两份不同坐标系的经纬度坐标数据统一坐标系。[coordtransform.py](https://github.com/JohnsonKlose/kfcmatching/blob/master/coordsimilarity/coordtransform.py)文件中包含了多种坐标系的转换方法，本项目运用bd09转换gcj02的方法，统一将百度地图坐标数据转换为高德地图坐标数据，转换方法如下所示：  
        ```
        x_pi = 3.14159265358979324 * 3000.0 / 180.0
        def bd09togcj02(bd_lon, bd_lat):
            """
            百度坐标系(BD-09)转火星坐标系(GCJ-02)
            百度——>谷歌、高德
            :param bd_lat:百度坐标纬度
            :param bd_lon:百度坐标经度
            :return:转换后的坐标列表形式
            """
            x = bd_lon - 0.0065
            y = bd_lat - 0.006
            z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
            theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
            gg_lng = z * math.cos(theta)
            gg_lat = z * math.sin(theta)
            return [gg_lng, gg_lat]
        ```
    3.  
        计算空间相似性的基本思路是距离越近得分越高，距离越远得分越低，距离高于某一阈值则全部为0。计算函数如下所示：  
        ```
        threshold = 0.01381770
        def coordmodeling(coord1, coord2):
            """
            计算空间相似性的方法
            """
            distance = math.sqrt((float(coord1[0])-float(coord2[0]))**2 + (float(coord1[1])-float(coord2[1]))**2)
            if distance >= threshold:
                return 0
            else:
                return 1 - distance/threshold
        ```  
        最后可以得到一个归一化的相似性结果，取值范围在0到1之间。  
        
- **语义相似性：**  
    语义相似性描述的是两条数据属性内容的相似性程度，本项目选择地址信息作为语义特征，通过给予jieba和gensim库构建文本相似性计算模型来计算两条数据之间的语义相似性，具体方法如下：  
    1. 
        自然语言处理首先要做的就是分词，这里使用最常见的结巴分词。分词做完之后还需要去停止词，去除标点符号、连词、助词等词性的词，同时再去除属于停止词库中的词，具体方法如下：  
        ```
        import jieba.posseg as pseg
        import codecs
        # 这里是停止词文件的地址，可替换成自己的停止词文件
        stopword_file = "/Users/yifengjiao/PycharmProjects/scrapDemo/dbcomments/stopwords.txt"
        stopwords = codecs.open(stopword_file, 'r', encoding='utf8').readlines()
        stopwords = [w.strip() for w in stopwords]
        stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
        """
        分词并去停止词
        """
        def cutwords(text):
            result = []
            words = pseg.cut(text)
            for word, flag in words:
                if flag not in stop_flag and word not in stopwords:
                    result.append(word)
            return result
        ```  
    2.     
        分词完成后，根据分词的结果，建立词袋模型。词袋模型是将分词结果的词频通过向量表示。例如：  
        
        麦当劳（哈西万达店）南岗区中兴大道168号哈西万达广场步行街1069
        
        这样一个文本，分词后的结果是
        
        ['麦当劳', '哈西', '万达', '店', '南岗区', '中兴', '大道', '哈西', '万达', '广场', '步行街']
        
        对应建立词袋模型的结果为
        
        [(0, 1), (1, 2), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]
        
        具体实现如下：  
        ```
        # 构建词袋模型
        dictionary = corpora.Dictionary(corpus)
        doc_vectors = [dictionary.doc2bow(text) for text in corpus]
        ```
    3.  
        光有词袋模型还是不够的，通过TF-IDF模型，对不同的词引入不同的权重。TF表示词频，IDF表示逆文档频率，即一个词在文本中所出现的频率倒数，表达的意思是一个词在某文本中出现的越多，在其他文本中出现的越少，则这个词能很好地反映这篇文本的内容，权重就越大。  
        TF-IDF模型的结果是与词袋模型相同维度的向量，只是把词频换成了对应TF*IDF的值。  
        ![词频TF](http://oswrmk9hd.bkt.clouddn.com/TF.png)  
        ![逆文档频率](http://oswrmk9hd.bkt.clouddn.com/IDF.png)  
        具体实现如下：  
        ```
        tfidf = models.TfidfModel(doc_vectors)
        tfidf_vectors = tfidf[doc_vectors]
        ```
        通过文本相似性模型计算，最后可以得到一个0到1的语义相似性结果，用于判断两条数据的地址内容相似度程度高低。  
        
- **匹配结果计算：**  
    获得了空间相似性和语义相似性的结果，如何获得最终的匹配结果呢？可以通过简单的设立阈值的方法判断，例如设立阈值为0.5，如果两个相似性同时大于0.5，则将判断此相似性结果是匹配成功。显然，这样的方法比较粗糙，误差也比较大，因此需要建立一种自动化匹配模型，能够自动判断某条相似性结果是否是匹配成功。  
    

## 自动匹配模型
基于空间和语义相似性计算得到的值，可以将结果可视化出来，如下图所示。我先粗略的设置阈值区分部分匹配成功的点，再通过人工检查最终将匹配结果全部区分成了未匹配和匹配成功两份数据集，蓝色点是未匹配的集合，红色为匹配成功的集合。  
    ![match_result](http://oswrmk9hd.bkt.clouddn.com/match_result.png)
    分析上图不难看出，我们可以在二维平面找到一个函数，将未匹配和匹配成功的点分开，找到了这个函数，也就解决了我们自动得出匹配结果的问题。这也等同于解决一个二分类问题，可以通过机器学习的分类模型来完成这个工作。本项目基于scikit-learn库，实现了SVM和LR两种分类模型。
- **支持向量机(Support Vector Machine)**  
    支持向量机简称SVM，它的目标是寻找区分两类的超平面，使边际最大(网络上对于SVM的介绍很多，详细介绍可自行查阅，这里推荐一篇博文仅供参考：[点击阅读](https://www.cnblogs.com/harvey888/p/5852687.html))。通过上图可以看出，我们可以把该匹配问题看作是线性分割问题，但也有少数蓝色点夹杂在红色中，也就是说仅有少数点线性不可分，因此更准确的，这个问题应该是“近似线性可分”。对于这类问题的处理，可通过引入松弛变量和惩罚因子来消除这些点的影响。具体请参考：[点击阅读](http://blog.csdn.net/qll125596718/article/details/6910921)  
    具体实现如下：  
    ```
    clf = svm.SVC(kernel='linear', class_weight={1:5})
    clf.fit(X, y)
    ```  
    kernel是选择核函数，这里解决线性问题因此选择linear，class_weight是为了解决数据不平衡问题，即正例反例数量不一致。给数量少的类设置更大的惩罚因子，可以一定程度消除数据不平衡带来的结果偏斜现象。  
    SVM训练得出的分类线性函数如下图所示，同时配上未加class_weight参数所得的分类平面以对比优化效果。  
    ![svm_classification](http://oswrmk9hd.bkt.clouddn.com/svm_classification.png)
    
- 对数几率回归(Logistic Regression)
    Logistic Regression虽然名字是回归，但本质是一个分类器，具体概念这里就不赘述了，具体可以参考周志华教授《机器学习》一书第3章第3节的内容，这里我们只需要知道和SVM一样，通过对数几率回归模型的学习，我们也可以学得一个线性函数可以分割未匹配和匹配成功两类数据。具体实现如下：  
    ```
    regr_optimize = linear_model.LogisticRegression(class_weight={1:5})
    regr_origin.fit(X, y)
    ```  
    LR训练得到的分类线性函数如下图所示，同样也展示提供class_weight参数和未提供class_weight参数两个模型学习的结果对比。  
    ![lr_classification](http://oswrmk9hd.bkt.clouddn.com/lr_classification.png)  
    **Tips:** Logistic Regression网络上很多人译为“逻辑回归”，但其实跟“逻辑”一点关系都没有，loic才可以译为“逻辑”。周志华老师将Logistic Regression译为“对数几率回归”我认为是相对准确的译法。  
    
## 展望
- 在计算语义相似性的过程中，出现了一些成功匹配但语义相似度很低的情况，考虑有几种情况造成：第一可能是不同数据源对于门店地址的语义表达不同；第二可能是分词的结果本身就不是很好，造成计算相似性的结果较差；第三可能是语义内容较少，只加入了地址信息，造成结果随机性提升。
- 本项目假设了该匹配问题线性可分，未来可以假设该匹配问题线性不可分，尝试通过非线性函数划分正例和反例，比如在SVM中使用非线性核函数等。
- 两个学习模型的泛化性能还需要其他数据来验证，未来可以爬取麦当劳数据来验证模型是否可以真正实现自动化匹配。  

## 联系我们
E-mail: 535848615@qq.com  
GitHub主页: [https://github.com/JohnsonKlose](https://github.com/JohnsonKlose)  
博客园: [http://www.cnblogs.com/KloseJiao/](http://www.cnblogs.com/KloseJiao/)  
喜欢的朋友们可以加个star，也欢迎留言和邮件与我交流！