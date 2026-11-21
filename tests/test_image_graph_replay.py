"""仅检查图节点融合、映射与邻居统计；不需要模型训练。"""
import unittest
import numpy as np
from experiments.image_graph_replay import fuse,broadcast_labels,pool_view_scores,neighbor_statistics,CachedDataset

class ImageGraphTests(unittest.TestCase):
    """原图身份由增强索引决定，不依赖真值或置信度阈值。"""
    def test_fusion_order(self):
        """两张图的视图只能在各自块内算术平均，保持float32。"""
        x=np.arange(64,dtype=np.float32).reshape(16,4);p=np.array([[1,0]]*8+[[0,1]]*8,dtype=np.float32)
        a,b=fuse(x,p);self.assertTrue(np.array_equal(a,np.stack([x[:8].mean(0),x[8:].mean(0)])));self.assertTrue(np.array_equal(b,[[1,0],[0,1]]));self.assertEqual(a.dtype,np.float32)
    def test_broadcast(self):
        """一个原图类别必须恰好映射回8个原位置。"""
        self.assertTrue(np.array_equal(broadcast_labels([3,1]),[3]*8+[1]*8))
    def test_neighbor_self(self):
        """把自身放在第二列，验证同图非自环比例不会误判。"""
        idx=np.array([[(i//8)*8+(i+1)%8,i,(i+8)%16] for i in range(16)]);r=neighbor_statistics(idx,np.ones((16,3)))
        self.assertEqual(r['self_count'],16);self.assertEqual(r['other_same_image_slot_fraction'],.5);self.assertEqual(r['same_image_weight_fraction'],.5);self.assertEqual(r['distinct_other_images_per_view']['mean'],1)
    def test_score_pool_not_vote(self):
        """分数均值控制不同于多数投票，避免混淆两个机制。"""
        scores=np.array([[0,.1]]*7+[[1,0]]+[[0,1]]*8,dtype=np.float16)
        self.assertTrue(np.array_equal(pool_view_scores(scores),[0,1]))
    def test_invalid(self):
        """不完整的视图块不能静默跨原图混合。"""
        with self.assertRaises(AssertionError):fuse(np.zeros((15,4)),np.zeros((15,2)))
    def test_fresh_dataset(self):
        """每次新建容器，硬标签回写不改变缓存原始概率。"""
        x=np.zeros((8,2),np.float32);p=np.full((8,2),.5,np.float32);d=CachedDataset(x,p);d.set_pseudo_labels(range(8),np.tile([1,0],(8,1)));self.assertTrue(np.array_equal(p,np.full((8,2),.5)));self.assertTrue(np.array_equal(CachedDataset(x,p).pseudo_labels,p))

if __name__=='__main__':unittest.main()
