# Introduction

Dataset Features
User ID, item ID, category ID, behavior type, timestamp

UserBehavior.csv

We random select about 1 million users who have behaviors including click, purchase, adding item to shopping cart and item favoring during November 25 to December 03, 2017. The dataset is organized in a very similar form to MovieLens-20M, i.e., each line represents a specific user-item interaction, which consists of user ID, item ID, item's category ID, behavior type and timestamp, separated by commas. The detailed descriptions of each field are as follows:
Field 	Explanation
User ID 	An integer, the serialized ID that represents a user
Item ID 	An integer, the serialized ID that represents an item
Category ID 	An integer, the serialized ID that represents the category which the corresponding item belongs to
Behavior type 	A string, enum-type from ('pv', 'buy', 'cart', 'fav')
Timestamp 	An integer, the timestamp of the behavior
Note that the dataset contains 4 different types of behaviors, they are
Behavior 	Explanation
pv 	Page view of an item's detail page, equivalent to an item click
buy 	Purchase an item
cart 	Add an item to shopping cart
fav 	Favor an item
Dimensions of the dataset are
Dimension 	Number
# of users 	987,994
# of items 	4,162,024
# of categories 	9,439
# of interactions 	100,150,807

Citations

1. Han Z, Xiang L, Pengye Z, et al. Learning Tree-based Deep Model for Recommender Systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
2. Han Z, Daqing C, Ziru X, et al. Joint Optimization of Tree-based Index and Deep Model for Recommender Systems. arXiv:1902.07565.