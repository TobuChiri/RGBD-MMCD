# NYUDepthv2 数据集类别映射 (40类)

## 类别索引与名称对应表

| 类别索引 | 类别名称 (英文) | 类别名称 (中文) |
|---------|----------------|----------------|
| 0 | wall | 墙 |
| 1 | floor | 地板 |
| 2 | cabinet | 橱柜 |
| 3 | bed | 床 |
| 4 | chair | 椅子 |
| 5 | sofa | 沙发 |
| 6 | table | 桌子 |
| 7 | door | 门 |
| 8 | window | 窗户 |
| 9 | bookshelf | 书架 |
| 10 | picture | 画/图片 |
| 11 | counter | 柜台/台面 |
| 12 | blinds | 百叶窗 |
| 13 | desk | 书桌 |
| 14 | shelves | 架子 |
| 15 | curtain | 窗帘 |
| 16 | dresser | 梳妆台 |
| 17 | pillow | 枕头 |
| 18 | mirror | 镜子 |
| 19 | floor mat | 地垫 |
| 20 | clothes | 衣服 |
| 21 | ceiling | 天花板 |
| 22 | books | 书籍 |
| 23 | refridgerator | 冰箱 |
| 24 | television | 电视 |
| 25 | paper | 纸张 |
| 26 | towel | 毛巾 |
| 27 | shower curtain | 浴帘 |
| 28 | box | 盒子 |
| 29 | whiteboard | 白板 |
| 30 | person | 人 |
| 31 | night stand | 床头柜 |
| 32 | toilet | 马桶 |
| 33 | sink | 水槽 |
| 34 | lamp | 灯 |
| 35 | bathtub | 浴缸 |
| 36 | bag | 包 |
| 37 | otherstructure | 其他结构 |
| 38 | otherfurniture | 其他家具 |
| 39 | otherprop | 其他道具 |

## 使用说明

在Grad-CAM可视化中：
- `target_class=0` 表示关注"墙"这个类别
- `target_class=4` 表示关注"椅子"这个类别
- `target_class=30` 表示关注"人"这个类别
- 以此类推...

## 常见室内场景类别

### 结构类 (0-2, 7-8, 12, 15, 21, 27, 37)
- 0: wall (墙)
- 1: floor (地板)
- 7: door (门)
- 8: window (窗户)
- 12: blinds (百叶窗)
- 15: curtain (窗帘)
- 21: ceiling (天花板)
- 27: shower curtain (浴帘)
- 37: otherstructure (其他结构)

### 家具类 (2-6, 9, 13-16, 23-24, 31-33, 35, 38)
- 2: cabinet (橱柜)
- 3: bed (床)
- 4: chair (椅子)
- 5: sofa (沙发)
- 6: table (桌子)
- 9: bookshelf (书架)
- 13: desk (书桌)
- 14: shelves (架子)
- 16: dresser (梳妆台)
- 23: refridgerator (冰箱)
- 24: television (电视)
- 31: night stand (床头柜)
- 32: toilet (马桶)
- 33: sink (水槽)
- 35: bathtub (浴缸)
- 38: otherfurniture (其他家具)

### 物品类 (10-11, 17-20, 22, 25-26, 28-30, 34, 36, 39)
- 10: picture (画/图片)
- 11: counter (柜台/台面)
- 17: pillow (枕头)
- 18: mirror (镜子)
- 19: floor mat (地垫)
- 20: clothes (衣服)
- 22: books (书籍)
- 25: paper (纸张)
- 26: towel (毛巾)
- 28: box (盒子)
- 29: whiteboard (白板)
- 30: person (人)
- 34: lamp (灯)
- 36: bag (包)
- 39: otherprop (其他道具)