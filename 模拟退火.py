import random
import math
from typing import List, Tuple, Dict, Set

class CuttingStockOptimizer:
    """切割优化器类"""
    
    class Config:
        """配置类，集中管理所有参数"""
        # 物理约束
        BLOCK_LENGTH = 3000
        ISOLATION_LENGTH = 5
        
        # 时间约束
        TIME_LIMIT_4 = {4, 6, 9, 12}
        TIME_LIMIT_6 = {2, 10, 16, 18}
        
        # 模拟退火参数
        INITIAL_TEMP = 1000
        FINAL_TEMP = 1
        COOLING_RATE = 0.95
        
        # 遗传算法参数
        POPULATION_SIZE = 300
        MAX_GENERATIONS = 300
        ELITE_RATE = 0.1
        CROSSOVER_RATE = 0.8
        TOURNAMENT_SIZE = 3
        
        # 评分权重
        BLOCK_WEIGHT = 10000
        PATTERN_WEIGHT = 5000
        WASTE_WEIGHT = 0.1

    def __init__(self, lengths: List[int], quantities: List[int]):
        """初始化优化器"""
        self.lengths = lengths
        self.quantities = quantities
        self.best_solution = None
        self.best_metrics = None

    def optimize(self) -> Tuple[List[List[int]], Dict]:
        """主优化流程"""
        # 第一阶段：模拟退火
        initial_solution = self._simulated_annealing()
        
        # 第二阶段：遗传算法
        final_solution = self._genetic_algorithm(initial_solution)
        
        # 计算最终指标
        self.best_solution = final_solution
        self.best_metrics = self._calculate_metrics(final_solution)
        
        return self.best_solution, self.best_metrics

    def _simulated_annealing(self) -> List[List[int]]:
        """模拟退火算法"""
        current = self._generate_initial_solution()
        best = current
        temp = self.Config.INITIAL_TEMP
        
        while temp > self.Config.FINAL_TEMP:
            new = self._get_neighbor(current)
            delta = self._evaluate(new) - self._evaluate(current)
            
            if delta < 0 or random.random() < math.exp(-delta/temp):
                current = new
                if self._evaluate(new) < self._evaluate(best):
                    best = new
                    
            temp *= self.Config.COOLING_RATE
            
        return best

    def _genetic_algorithm(self, initial_solution: List[List[int]]) -> List[List[int]]:
        """遗传算法"""
        population = self._initialize_population(initial_solution)
        best_solution = initial_solution
        generations_without_improvement = 0
        
        for generation in range(self.Config.MAX_GENERATIONS):
            # 评估种群
            fitness_scores = [self._evaluate(solution) for solution in population]
            current_best = population[fitness_scores.index(min(fitness_scores))]
            
            # 更新最佳解
            if self._evaluate(current_best) < self._evaluate(best_solution):
                best_solution = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # 提前终止检查
            if generations_without_improvement > 50:
                break
                
            # 进化种群
            population = self._evolve_population(population, fitness_scores)
            
            # 局部搜索
            if generation % 10 == 0:
                best_solution = self._local_search(best_solution)
        
        return best_solution

    # 解的生成和评估
    def _generate_initial_solution(self) -> List[List[int]]:
        """生成初始解"""
        solution = []
        remaining = self.quantities.copy()
        
        while sum(remaining) > 0:
            pattern = self._generate_pattern(remaining)
            if not pattern:
                break
            solution.append(pattern)
            
        return solution

    def _evaluate(self, solution: List[List[int]]) -> float:
        """评估解的质量"""
        if not self._is_valid_solution(solution):
            return float('inf')
            
        blocks = len(solution)
        patterns = len(set(tuple(p) for p in solution))
        waste = sum(self._calculate_waste(p) for p in solution)
        
        return (blocks * self.Config.BLOCK_WEIGHT + 
                patterns * self.Config.PATTERN_WEIGHT + 
                waste * self.Config.WASTE_WEIGHT)

    # 解的验证和计算
    def _is_valid_solution(self, solution: List[List[int]]) -> bool:
        """验证解的可行性"""
        if not solution:
            return False
            
        # 验证每个模式
        if not all(self._is_valid_pattern(p) for p in solution):
            return False
            
        # 验证需求满足
        counts = self._count_pieces(solution)
        return all(counts[i] >= self.quantities[i] for i in range(len(self.quantities)))

    def _is_valid_pattern(self, pattern: List[int]) -> bool:
        """验证切割模式的可行性"""
        if not pattern:
            return False
        total_length = sum(self.lengths[i] for i in pattern)
        total_length += (len(pattern) - 1) * self.Config.ISOLATION_LENGTH
        return total_length <= self.Config.BLOCK_LENGTH

    def _calculate_waste(self, pattern: List[int]) -> float:
        """计算废料长度"""
        used_length = sum(self.lengths[i] for i in pattern)
        used_length += (len(pattern) - 1) * self.Config.ISOLATION_LENGTH
        return self.Config.BLOCK_LENGTH - used_length

    # 种群演化操作
    def _initialize_population(self, initial_solution: List[List[int]]) -> List[List[List[int]]]:
        """初始化种群"""
        population = [initial_solution]
        while len(population) < self.Config.POPULATION_SIZE:
            new_solution = self._get_neighbor(random.choice(population))
            if self._is_valid_solution(new_solution):
                population.append(new_solution)
        return population

    def _evolve_population(self, population: List[List[List[int]]], 
                          fitness_scores: List[float]) -> List[List[List[int]]]:
        """种群进化"""
        elite_size = int(self.Config.POPULATION_SIZE * self.Config.ELITE_RATE)
        elite = sorted(zip(fitness_scores, population))[:elite_size]
        elite = [solution for _, solution in elite]
        
        new_population = elite.copy()
        while len(new_population) < self.Config.POPULATION_SIZE:
            if random.random() < self.Config.CROSSOVER_RATE:
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)
                child = self._crossover(parent1, parent2)
            else:
                child = self._mutate(random.choice(elite))
                
            if self._is_valid_solution(child):
                new_population.append(child)
                
        return new_population

    # 辅助方法
    def _calculate_metrics(self, solution: List[List[int]]) -> Dict:
        """计算解的各项指标"""
        return {
            'total_blocks': len(solution),
            'pattern_changes': len(set(tuple(p) for p in solution)) - 1,
            'total_waste': sum(self._calculate_waste(p) for p in solution),
            'time_4_blocks': self._count_time_blocks(solution, self.Config.TIME_LIMIT_4),
            'time_6_blocks': self._count_time_blocks(solution, self.Config.TIME_LIMIT_6)
        }

    def _count_time_blocks(self, solution: List[List[int]], time_set: Set[int]) -> int:
        """计算时间约束内的场地数"""
        return sum(1 for pattern in solution if any(i+1 in time_set for i in pattern))

    def _count_pieces(self, solution: List[List[int]]) -> List[int]:
        """统计各规格数量"""
        counts = [0] * len(self.lengths)
        for pattern in solution:
            for piece in pattern:
                counts[piece] += 1
        return counts

    def _generate_pattern(self, remaining: List[int]) -> List[int]:
        """生成单个切割模式"""
        pattern = []
        current_length = 0
        
        # 按长度降序排列可选的规格
        available_pieces = [(i, self.lengths[i]) for i in range(len(self.lengths)) if remaining[i] > 0]
        available_pieces.sort(key=lambda x: x[1], reverse=True)
        
        # 贪心策略填充
        for piece_index, piece_length in available_pieces:
            while remaining[piece_index] > 0:
                # 计算添加新件后的总长度（包括隔离带）
                new_length = current_length
                if pattern:  # 如果不是第一件，需要加上隔离带
                    new_length += self.Config.ISOLATION_LENGTH
                new_length += piece_length
                
                # 检查是否超出限制
                if new_length > self.Config.BLOCK_LENGTH:
                    break
                    
                # 添加新件
                pattern.append(piece_index)
                current_length = new_length
                remaining[piece_index] -= 1
        
        return pattern

    def _get_neighbor(self, solution: List[List[int]]) -> List[List[int]]:
        """生成邻域解"""
        new_solution = [pattern.copy() for pattern in solution]
        
        # 随机选择操作类型
        operation = random.choice(['swap', 'move', 'merge', 'split'])
        
        if operation == 'swap' and len(new_solution) >= 2:
            # 交换两个模式中的部件
            i, j = random.sample(range(len(new_solution)), 2)
            if new_solution[i] and new_solution[j]:
                pos1 = random.randrange(len(new_solution[i]))
                pos2 = random.randrange(len(new_solution[j]))
                new_solution[i][pos1], new_solution[j][pos2] = new_solution[j][pos2], new_solution[i][pos1]
                
        elif operation == 'move' and len(new_solution) >= 2:
            # 移动一个部件到另一个模式
            i, j = random.sample(range(len(new_solution)), 2)
            if new_solution[i]:
                piece = new_solution[i].pop(random.randrange(len(new_solution[i])))
                new_solution[j].append(piece)
                
        elif operation == 'merge' and len(new_solution) >= 2:
            # 尝试合并两个模式
            i, j = random.sample(range(len(new_solution)), 2)
            merged = new_solution[i] + new_solution[j]
            if self._is_valid_pattern(merged):
                new_solution = [p for k, p in enumerate(new_solution) if k not in (i, j)]
                new_solution.append(merged)
                
        elif operation == 'split' and new_solution:
            # 分割一个模式
            i = random.randrange(len(new_solution))
            if len(new_solution[i]) > 1:
                split_point = random.randrange(1, len(new_solution[i]))
                pattern1 = new_solution[i][:split_point]
                pattern2 = new_solution[i][split_point:]
                if self._is_valid_pattern(pattern1) and self._is_valid_pattern(pattern2):
                    new_solution[i] = pattern1
                    new_solution.append(pattern2)
        
        # 移除空模式并验证可行性
        new_solution = [pattern for pattern in new_solution if pattern]
        return new_solution if self._is_valid_solution(new_solution) else solution

    def _tournament_select(self, population: List[List[List[int]]], fitness_scores: List[float]) -> List[List[int]]:
        """锦标赛选择"""
        tournament_size = self.Config.TOURNAMENT_SIZE
        selected_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in selected_indices]
        winner_index = selected_indices[tournament_fitness.index(min(tournament_fitness))]
        return population[winner_index]

    def _crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        """交叉操作"""
        if not parent1 or not parent2:
            return parent1 if parent1 else parent2
            
        # 随机选择交叉点
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        
        # 生成子代
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        return child

    def _mutate(self, solution: List[List[int]]) -> List[List[int]]:
        """变异操作"""
        return self._get_neighbor(solution)

    def _local_search(self, solution: List[List[int]]) -> List[List[int]]:
        """局部搜索优化"""
        improved = True
        best_solution = solution
        best_score = self._evaluate(solution)
        
        while improved:
            improved = False
            
            # 尝试所有可能的局部改进
            for i in range(len(best_solution)):
                # 尝试重排当前模式
                new_pattern = sorted(best_solution[i], key=lambda x: self.lengths[x], reverse=True)
                if new_pattern != best_solution[i]:
                    new_solution = best_solution.copy()
                    new_solution[i] = new_pattern
                    new_score = self._evaluate(new_solution)
                    
                    if new_score < best_score:
                        best_solution = new_solution
                        best_score = new_score
                        improved = True
                
                # 尝试与其他模式合并
                for j in range(i + 1, len(best_solution)):
                    merged = best_solution[i] + best_solution[j]
                    if self._is_valid_pattern(merged):
                        new_solution = [p for k, p in enumerate(best_solution) if k not in (i, j)]
                        new_solution.append(merged)
                        new_score = self._evaluate(new_solution)
                        
                        if new_score < best_score:
                            best_solution = new_solution
                            best_score = new_score
                            improved = True
                            break
        
        return best_solution

def main():
    """主函数"""
    # 测试数据
    lengths = [1532, 1477, 1285, 1232, 1046, 882, 830, 766, 732, 600, 582, 578, 455, 415, 405, 328, 255, 200]
    quantities = [104, 38, 60, 4, 4, 150, 30, 4, 34, 80, 196, 8, 52, 8, 136, 4, 292, 212]
    
    # 创建优化器并运行
    optimizer = CuttingStockOptimizer(lengths, quantities)
    solution, metrics = optimizer.optimize()
    
    # 输出结果
    print_results(solution, metrics, optimizer)

def print_results(solution: List[List[int]], metrics: Dict, optimizer: CuttingStockOptimizer):
    """打印结果"""
    print("=" * 50)
    print("最优切割方案详情：")
    print("=" * 50)
    
    # 打印主要指标
    print("1. 主要指标：")
    print(f"   - 需要大块场地数量：{metrics['total_blocks']} 块")
    print(f"   - 划分方式改变次数：{metrics['pattern_changes']} 次")
    print(f"   - 总废料长度：{metrics['total_waste']:.2f} 单位")
    
    # 打印具体方案
    print("\n2. 具体切割方案：")
    for i, pattern in enumerate(solution, 1):
        pieces = [f"{optimizer.lengths[item]}({item+1}号)" for item in pattern]
        total_length = (sum(optimizer.lengths[item] for item in pattern) + 
                       (len(pattern)-1) * optimizer.Config.ISOLATION_LENGTH)
        waste = optimizer.Config.BLOCK_LENGTH - total_length
        
        print(f"   第{i}块场地: {' + '.join(pieces)}")
        print(f"   使用长度: {total_length}, 剩余: {waste:.2f}")
        print(f"   {'-' * 40}")

if __name__ == "__main__":
    main()