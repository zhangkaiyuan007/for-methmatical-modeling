import random
import math
import numpy as np
from typing import List, Tuple

# 常量定义
LENGTHS = [1532, 1477, 1285, 1232, 1046, 882, 830, 766, 732, 600, 582, 578, 455, 415, 405, 328, 255, 200]
QUANTITIES = [104, 38, 60, 4, 4, 150, 30, 4, 34, 80, 196, 8, 52, 8, 136, 4, 292, 212]
BLOCK_LENGTH = 3000
ISOLATION_LENGTH = 5
TIME_LIMIT_4 = {4, 6, 9, 12}  # 4个时间单位内完成的场地编号
TIME_LIMIT_6 = {2, 10, 16, 18}  # 6个时间单位内完成的场地编号

class Solution:
    def __init__(self):
        self.best_solution = None
        self.best_score = (float('inf'), float('inf'), float('inf'))
    
    def calculate_waste(self, pattern: List[int]) -> float:
        """计算一个划分方案的废料长度"""
        current_length = 0
        isolations = len(pattern) - 1
        
        for item in pattern:
            current_length += LENGTHS[item]
        
        current_length += isolations * ISOLATION_LENGTH
        
        if current_length > BLOCK_LENGTH:
            return float('inf')
        
        return BLOCK_LENGTH - current_length

    def evaluate_solution(self, solution: List[List[int]]) -> Tuple[int, int, float]:
        """评估解决方案,返回(大块数量,改变次数,总废料)"""
        total_blocks = len(solution)
        pattern_changes = len(set(tuple(pattern) for pattern in solution)) - 1
        total_waste = sum(self.calculate_waste(pattern) for pattern in solution)
        
        return total_blocks, pattern_changes, total_waste

    def is_valid_solution(self, solution: List[List[int]]) -> bool:
        """检查解决方案是否满足所有约束"""
        # 统计每种规格的数量
        counts = [0] * len(LENGTHS)
        for pattern in solution:
            for item in pattern:
                counts[item] += 1
                
        # 检查是否满足需求数量
        for i in range(len(LENGTHS)):
            if counts[i] < QUANTITIES[i]:
                return False
                
        return True

    def simulated_annealing(self):
        """模拟退火算法主体"""
        T = 1000  # 初始温度
        T_min = 0.1  # 最小温度
        alpha = 0.98  # 降温系数
        
        # 初始解
        current_solution = self.generate_initial_solution()
        current_score = self.evaluate_solution(current_solution)
        
        # 初始化最佳解
        self.best_solution = current_solution
        self.best_score = current_score
        
        while T > T_min:
            for _ in range(100):  # 每个温度进行100次迭代
                new_solution = self.get_neighbor(current_solution)
                if not self.is_valid_solution(new_solution):
                    continue
                    
                new_score = self.evaluate_solution(new_solution)
                
                # 计算接受概率
                delta = (new_score[0] * 10000 + new_score[1] * 100 + new_score[2]) - \
                        (current_score[0] * 10000 + current_score[1] * 100 + current_score[2])
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_solution = new_solution
                    current_score = new_score
                    
                    # 更新最佳解
                    current_total = current_score[0] * 10000 + current_score[1] * 100 + current_score[2]
                    best_total = self.best_score[0] * 10000 + self.best_score[1] * 100 + self.best_score[2]
                    
                    if current_total < best_total:
                        self.best_solution = current_solution.copy()
                        self.best_score = current_score
            
            T *= alpha
            
        return self.best_solution, self.best_score

    def generate_initial_solution(self) -> List[List[int]]:
        """生成初始解"""
        solution = []
        remaining = QUANTITIES.copy()
        
        while sum(remaining) > 0:
            pattern = []
            current_length = 0
            
            for i in range(len(LENGTHS)):
                while remaining[i] > 0:
                    if current_length + LENGTHS[i] + (len(pattern) * ISOLATION_LENGTH) <= BLOCK_LENGTH:
                        pattern.append(i)
                        remaining[i] -= 1
                        current_length += LENGTHS[i]
                    else:
                        break
                        
            if pattern:
                solution.append(pattern)
                
        return solution

    def get_neighbor(self, solution: List[List[int]]) -> List[List[int]]:
        """生成邻域解"""
        new_solution = [pattern.copy() for pattern in solution]
        
        # 随机选择操作类型
        operation = random.choice(['swap', 'move', 'merge_split'])
        
        if operation == 'swap':
            # 交换两个模式中的元素
            if len(new_solution) >= 2:
                i, j = random.sample(range(len(new_solution)), 2)
                if new_solution[i] and new_solution[j]:
                    pos1 = random.randrange(len(new_solution[i]))
                    pos2 = random.randrange(len(new_solution[j]))
                    new_solution[i][pos1], new_solution[j][pos2] = \
                    new_solution[j][pos2], new_solution[i][pos1]
                    
        elif operation == 'move':
            # 移动一个元素到另一个模式
            if len(new_solution) >= 2:
                i, j = random.sample(range(len(new_solution)), 2)
                if new_solution[i]:
                    pos = random.randrange(len(new_solution[i]))
                    item = new_solution[i].pop(pos)
                    new_solution[j].append(item)
                    
        else:  # merge_split
            # 合并两个模式并重新分配
            if len(new_solution) >= 2:
                i, j = random.sample(range(len(new_solution)), 2)
                merged = new_solution[i] + new_solution[j]
                random.shuffle(merged)
                split_point = random.randrange(1, len(merged))
                new_solution[i] = merged[:split_point]
                new_solution[j] = merged[split_point:]
                
        return new_solution
    
    # 继续上一段代码
class TimeConstrainedOptimizer:
    def __init__(self, initial_solution):
        self.initial_solution = initial_solution
        self.max_blocks_per_time = 35
        self.best_global_solution = None
        self.best_global_score = float('inf')
    
    def optimize_with_time_constraints(self):
        """考虑时间约束的多目标优化"""
        # 将任务按时间要求分组
        time_4_tasks = self.get_time_constrained_tasks(TIME_LIMIT_4)
        time_6_tasks = self.get_time_constrained_tasks(TIME_LIMIT_6)
        remaining_tasks = self.get_remaining_tasks(TIME_LIMIT_4 | TIME_LIMIT_6)
        
        # 分阶段优化
        solution_4 = self.optimize_time_window(time_4_tasks, 4)
        solution_6 = self.optimize_time_window(time_6_tasks, 6)
        solution_remaining = self.optimize_remaining(remaining_tasks)
        
        return self.merge_solutions(solution_4, solution_6, solution_remaining)
    
    def get_time_constrained_tasks(self, task_indices):
        """获取特定时间约束的任务"""
        tasks = {}
        for idx in task_indices:
            idx = idx - 1  # 转换为0-based索引
            tasks[idx] = QUANTITIES[idx]
        return tasks
    
    def get_remaining_tasks(self, excluded_indices):
        """获取剩余任务"""
        tasks = {}
        for i in range(len(LENGTHS)):
            if i + 1 not in excluded_indices:
                tasks[i] = QUANTITIES[i]
        return tasks
    
    def optimize_remaining(self, tasks):
        """化剩余任务"""
        return self.optimize_time_window(tasks, float('inf'))
    
    def merge_solutions(self, solution_4, solution_6, solution_remaining):
        """合并各时间窗口的解决方案"""
        final_solution = []
        if solution_4:
            final_solution.extend(solution_4)
        if solution_6:
            final_solution.extend(solution_6)
        if solution_remaining:
            final_solution.extend(solution_remaining)
        return final_solution
    
    def calculate_waste(self, pattern: List[int]) -> float:
        """计算一个划分方案的废料长度"""
        if not pattern:
            return BLOCK_LENGTH
            
        current_length = 0
        isolations = len(pattern) - 1
        
        for item in pattern:
            current_length += LENGTHS[item]
        
        current_length += isolations * ISOLATION_LENGTH
        
        if current_length > BLOCK_LENGTH:
            return float('inf')
        
        return BLOCK_LENGTH - current_length
    
    def is_valid_pattern(self, pattern: List[int]) -> bool:
        """检查模式是否有效"""
        if not pattern:
            return False
            
        current_length = 0
        isolations = len(pattern) - 1
        
        for item in pattern:
            current_length += LENGTHS[item]
        current_length += isolations * ISOLATION_LENGTH
        
        return current_length <= BLOCK_LENGTH
    
    def optimize_time_window(self, tasks, time_limit):
        """优化特定时间窗口内的任务"""
        if not tasks:
            return []
            
        max_blocks = self.max_blocks_per_time * time_limit
        
        def objective_function(solution):
            if not solution or not self.is_valid_solution(solution, tasks):
                return float('inf')
                
            blocks_used = len(solution)
            if blocks_used > max_blocks:
                return float('inf')
                
            pattern_changes = len(set(tuple(pattern) for pattern in solution)) - 1
            total_waste = sum(self.calculate_waste(pattern) for pattern in solution)
            
            # 更新权重比例
            return (blocks_used * 10000 +  # 最高优先级：减少大块场地数量
                   pattern_changes * 5000 +  # 次高优先级：减少改变次数
                   total_waste * 0.1)  # 最低优先级：减少废料
        
        # 增加种群多样性
        population_size = 300
        generations = 300
        elite_size = int(population_size * 0.1)  # 保留10%的精英
        
        # 使用多个初始种群
        populations = []
        for _ in range(3):  # 创建3个不同的初始种群
            pop = self.initialize_population(tasks, population_size // 3)
            populations.extend(pop)
        
        population = populations
        best_solution = None
        best_fitness = float('inf')
        
        # 遗传算法迭代
        no_improvement_count = 0
        for generation in range(generations):
            # 评估适应度
            fitness_scores = [objective_function(solution) for solution in population]
            
            # 更新最佳解
            min_fitness_idx = fitness_scores.index(min(fitness_scores))
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_solution = [pattern.copy() for pattern in population[min_fitness_idx]]
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 每10代进行一次局部搜索
            if generation % 10 == 0 and best_solution:
                improved_solution = self.local_search(best_solution)
                improved_fitness = objective_function(improved_solution)
                if improved_fitness < best_fitness:
                    best_solution = improved_solution
                    best_fitness = improved_fitness
                    no_improvement_count = 0
            
            # 提前终止条件
            if no_improvement_count > 50:  # 增加容忍度
                break
            
            # 精英保留
            elite = []
            sorted_indices = sorted(range(len(fitness_scores)), 
                                  key=lambda k: fitness_scores[k])
            for idx in sorted_indices[:elite_size]:
                elite.append([pattern.copy() for pattern in population[idx]])
            
            # 选择和繁殖
            new_population = []
            new_population.extend(elite)
            
            # 动态调整变异率
            mutation_rate = 0.1 + (no_improvement_count / 100)  # 随着无改善代数增加而增加变异率
            mutation_rate = min(mutation_rate, 0.5)  # 限制最大变异率
            
            while len(new_population) < population_size:
                if random.random() < 0.8:  # 80%概率使用交叉
                    parent1 = self.tournament_select(population, fitness_scores)
                    parent2 = self.tournament_select(population, fitness_scores)
                    child = self.crossover(parent1, parent2)
                else:  # 20%概率直接复制精英并变异
                    child = random.choice(elite).copy()
                
                child = self.mutate(child, mutation_rate)
                if self.is_valid_solution(child, tasks):
                    new_population.append(child)
            
            population = new_population
            
            # 每50代注入新的随机解以维持多样性
            if generation % 50 == 0:
                num_random = int(population_size * 0.1)  # 注入10%的新随机解
                for _ in range(num_random):
                    random_sol = self.random_solution(tasks)
                    if random_sol:
                        population[-1] = random_sol
        
        return best_solution
    
    def tournament_select(self, population, fitness_scores, tournament_size=3):
        """锦标赛选择"""
        tournament_idx = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_idx]
        winner_idx = tournament_idx[tournament_fitness.index(min(tournament_fitness))]
        return population[winner_idx]
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        if not parent1 or not parent2:
            return parent1 if parent1 else parent2
            
        crossover_point = random.randint(1, min(len(parent1), len(parent2)))
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def initialize_population(self, tasks, pop_size):
        """改进的初始化种群方法"""
        population = []
        
        # 使用不同的启发式方法生成初始解
        for _ in range(pop_size):
            if random.random() < 0.5:
                # 首适应方法
                solution = self.first_fit_solution(tasks)
            else:
                # 最佳适应方法
                solution = self.best_fit_solution(tasks)
            
            if solution and self.is_valid_solution(solution, tasks):
                population.append(solution)
        
        # 如果生成的解不够，使用随机方法补充
        while len(population) < pop_size:
            solution = self.random_solution(tasks)
            if solution and self.is_valid_solution(solution, tasks):
                population.append(solution)
        
        return population
    
    def first_fit_solution(self, tasks):
        """首适应算法生成解"""
        solution = []
        remaining = tasks.copy()
        
        while sum(remaining.values()) > 0:
            pattern = []
            current_length = 0
            
            for item_id in sorted(remaining.keys()):
                while remaining[item_id] > 0:
                    test_length = current_length + LENGTHS[item_id]
                    if len(pattern) > 0:
                        test_length += ISOLATION_LENGTH
                        
                    if test_length <= BLOCK_LENGTH:
                        pattern.append(item_id)
                        remaining[item_id] -= 1
                        current_length = test_length
                    else:
                        break
            
            if pattern:
                solution.append(pattern)
            else:
                break
        
        return solution if self.is_valid_solution(solution, tasks) else None
    
    def best_fit_solution(self, tasks):
        """最佳适应算法生成解"""
        solution = []
        remaining = tasks.copy()
        
        while sum(remaining.values()) > 0:
            pattern = []
            current_length = 0
            
            while True:
                best_item = None
                min_waste = BLOCK_LENGTH
                
                for item_id, quantity in remaining.items():
                    if quantity > 0:
                        test_length = current_length + LENGTHS[item_id]
                        if len(pattern) > 0:
                            test_length += ISOLATION_LENGTH
                            
                        if test_length <= BLOCK_LENGTH:
                            waste = BLOCK_LENGTH - test_length
                            if waste < min_waste:
                                min_waste = waste
                                best_item = item_id
                
                if best_item is None:
                    break
                    
                pattern.append(best_item)
                remaining[best_item] -= 1
                current_length += LENGTHS[best_item]
                if len(pattern) > 1:
                    current_length += ISOLATION_LENGTH
            
            if pattern:
                solution.append(pattern)
            else:
                break
        
        return solution if self.is_valid_solution(solution, tasks) else None
    
    def random_solution(self, tasks):
        """随机生成一个有效解"""
        solution = []
        remaining = tasks.copy()
        
        while sum(remaining.values()) > 0:
            pattern = []
            current_length = 0
            
            available_items = list(remaining.keys())
            random.shuffle(available_items)
            
            for item_id in available_items:
                if remaining[item_id] > 0:
                    test_length = current_length + LENGTHS[item_id]
                    if len(pattern) > 0:
                        test_length += ISOLATION_LENGTH
                        
                    if test_length <= BLOCK_LENGTH:
                        pattern.append(item_id)
                        remaining[item_id] -= 1
                        current_length = test_length
                
            if pattern:
                solution.append(pattern)
            else:
                if sum(remaining.values()) > 0:
                    continue
                break
        
        return solution if self.is_valid_solution(solution, tasks) else None
    
    def local_search(self, solution):
        """增强版局部搜索"""
        if not solution:
            return solution
            
        improved = True
        best_solution = [pattern.copy() for pattern in solution]
        
        while improved:
            improved = False
            
            # 尝试合并相邻模式
            for i in range(len(best_solution)-1):
                for j in range(i+1, len(best_solution)):
                    merged = best_solution[i] + best_solution[j]
                    if self.is_valid_pattern(merged):
                        new_solution = [p.copy() for p in best_solution if p != best_solution[i] and p != best_solution[j]]
                        new_solution.append(merged)
                        if len(new_solution) < len(best_solution):
                            best_solution = new_solution
                            improved = True
                            break
                if improved:
                    break
            
            # 尝试重新排列每个模式
            if not improved:
                for i in range(len(best_solution)):
                    pattern = best_solution[i]
                    best_waste = self.calculate_waste(pattern)
                    
                    # 尝试不同的排列
                    for _ in range(20):
                        new_pattern = pattern.copy()
                        random.shuffle(new_pattern)
                        new_waste = self.calculate_waste(new_pattern)
                        
                        if new_waste < best_waste:
                            best_solution[i] = new_pattern
                            improved = True
                            break
            
            # 尝试分割和重组
            if not improved:
                for i in range(len(best_solution)):
                    if len(best_solution[i]) > 2:
                        split_point = len(best_solution[i]) // 2
                        part1 = best_solution[i][:split_point]
                        part2 = best_solution[i][split_point:]
                        
                        if self.is_valid_pattern(part1) and self.is_valid_pattern(part2):
                            new_solution = [p.copy() for p in best_solution if p != best_solution[i]]
                            new_solution.extend([part1, part2])
                            
                            # 如果新解更好，接受它
                            if len(set(tuple(p) for p in new_solution)) < len(set(tuple(p) for p in best_solution)):
                                best_solution = new_solution
                                improved = True
                                break
        
        return best_solution
    
    def mutate(self, solution, mutation_rate=0.1):
        """变异操作，确保生成有效的解"""
        if random.random() > mutation_rate or not solution:
            return solution
        
        mutated = [pattern.copy() for pattern in solution]  # 深拷贝
        mutation_type = random.choice(['swap', 'split', 'merge'])
        
        max_attempts = 10  # 最大尝试次数
        for _ in range(max_attempts):
            temp_solution = [pattern.copy() for pattern in mutated]
            
            if mutation_type == 'swap' and len(temp_solution) >= 2:
                # 交换两个模式中的元素
                i, j = random.sample(range(len(temp_solution)), 2)
                if temp_solution[i] and temp_solution[j]:
                    pos1 = random.randrange(len(temp_solution[i]))
                    pos2 = random.randrange(len(temp_solution[j]))
                    temp_solution[i][pos1], temp_solution[j][pos2] = \
                        temp_solution[j][pos2], temp_solution[i][pos1]
                    
                    if (self.is_valid_pattern(temp_solution[i]) and 
                        self.is_valid_pattern(temp_solution[j])):
                        mutated = temp_solution
                        break
                    
            elif mutation_type == 'split' and len(temp_solution) > 0:
                # 分割一个模式
                i = random.randrange(len(temp_solution))
                if len(temp_solution[i]) > 1:
                    split_point = random.randrange(1, len(temp_solution[i]))
                    new_pattern = temp_solution[i][split_point:]
                    temp_solution[i] = temp_solution[i][:split_point]
                    
                    if (self.is_valid_pattern(temp_solution[i]) and 
                        self.is_valid_pattern(new_pattern)):
                        temp_solution.append(new_pattern)
                        mutated = temp_solution
                        break
                    
            elif mutation_type == 'merge' and len(temp_solution) >= 2:
                # 合并两个模式
                i, j = random.sample(range(len(temp_solution)), 2)
                merged = temp_solution[i] + temp_solution[j]
                
                if self.is_valid_pattern(merged):
                    new_solution = [p.copy() for p in temp_solution if p != temp_solution[i] and p != temp_solution[j]]
                    new_solution.append(merged)
                    mutated = new_solution
                    break
        
        return mutated

    def is_valid_solution(self, solution: List[List[int]], tasks: dict) -> bool:
        """检查解决方案是否满足所有约束"""
        if not solution:
            return False
            
        # 检查每个模式是否有效
        for pattern in solution:
            if not self.is_valid_pattern(pattern):
                return False
        
        # 检查是否满足需求数量
        counts = {}
        for pattern in solution:
            for item in pattern:
                counts[item] = counts.get(item, 0) + 1
        
        for task_id, required_quantity in tasks.items():
            if counts.get(task_id, 0) < required_quantity:
                return False
        
        return True

# 主程序运行
def main():
    # 第一阶段：使用模拟退火获取初始解
    sa_solver = Solution()
    initial_solution, initial_score = sa_solver.simulated_annealing()
    
    # 第二阶段：考虑时间约束的多目标优化
    time_optimizer = TimeConstrainedOptimizer(initial_solution)
    final_solution = time_optimizer.optimize_with_time_constraints()
    
    # 详细输出结果
    print("=" * 50)
    print("最优切割方案详情：")
    print("=" * 50)
    
    # 1. 基本指标
    blocks_count = len(final_solution)
    pattern_changes = len(set(tuple(pattern) for pattern in final_solution)) - 1
    total_waste = sum(sa_solver.calculate_waste(pattern) for pattern in final_solution)
    
    print(f"1. 主要指标：")
    print(f"   - 需要大块场地数量：{blocks_count} 块")
    print(f"   - 划分方式改变次数：{pattern_changes} 次")
    print(f"   - 总废料长度：{total_waste:.2f} 单位")
    
    # 2. 详细的切割方案
    print("\n2. 具体切割方案：")
    for i, pattern in enumerate(final_solution, 1):
        pieces = [f"{LENGTHS[item]}({item+1}号)" for item in pattern]
        total_length = sum(LENGTHS[item] for item in pattern) + (len(pattern)-1) * ISOLATION_LENGTH
        waste = BLOCK_LENGTH - total_length
        print(f"   第{i}块场地: {' + '.join(pieces)}")
        print(f"   使用长度: {total_length}, 剩余: {waste:.2f}")
        print(f"   {'-' * 40}")
    
    # 3. 时间约束验证
    print("\n3. 时间约束验证：")
    time_4_patterns = []
    time_6_patterns = []
    other_patterns = []
    
    for pattern in final_solution:
        if any(i in [x-1 for x in TIME_LIMIT_4] for i in pattern):
            time_4_patterns.append(pattern)
        elif any(i in [x-1 for x in TIME_LIMIT_6] for i in pattern):
            time_6_patterns.append(pattern)
        else:
            other_patterns.append(pattern)
    
    print(f"   4时间单位内完成的场地数：{len(time_4_patterns)}")
    print(f"   6时间单位内完成的场地数：{len(time_6_patterns)}")
    print(f"   其他场地数：{len(other_patterns)}")

if __name__ == "__main__":
    main()