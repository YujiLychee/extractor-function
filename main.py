import signal
import time
import threading
import os
import psutil
import logging

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("模型加载超时")

def init_extractor_background():
    global extractor, _initialization_status, _initialization_start_time

    with _initialization_lock:
        if _initialization_status != "pending":
            return

        _initialization_status = "loading"
        _initialization_start_time = time.time()
        logger.info("开始后台初始化...")

        try:
            # 检查可用内存
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            logger.info(f"可用内存: {available_gb:.1f}GB")
            
            # 如果内存不足，直接使用规则模式
            if available_gb < 1.5:
                logger.warning(f"内存不足({available_gb:.1f}GB)，使用规则模式")
                extractor = SmartNewsExtractor(use_bert=False, preload_db="property_translations.db")
                _initialization_status = "ready_fallback"
                load_time = time.time() - _initialization_start_time
                logger.info(f"规则模式初始化完成 (耗时: {load_time:.2f}秒)")
                return

            # 设置120秒超时
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)

            logger.info("尝试 BERT 模式初始化...")
            
            # 检查缓存目录
            cache_dir = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", "/app/cache"))
            logger.info(f"检查缓存目录: {cache_dir}")
            
            if os.path.exists(cache_dir):
                cache_files = os.listdir(cache_dir)
                logger.info(f"缓存目录内容: {len(cache_files)} 个文件/目录")
            else:
                logger.warning(f"缓存目录不存在: {cache_dir}")

            extractor = SmartNewsExtractor(
                use_bert=True,
                preload_db="property_translations.db"
            )

            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # 取消超时

            load_time = time.time() - _initialization_start_time
            logger.info(f"BERT 模式初始化成功 (耗时: {load_time:.2f}秒)")
            _initialization_status = "ready"

        except TimeoutException:
            logger.warning("BERT 模式加载超时，切换到规则模式")
            try:
                extractor = SmartNewsExtractor(use_bert=False, preload_db="property_translations.db")
                _initialization_status = "ready_fallback"
                load_time = time.time() - _initialization_start_time
                logger.info(f"规则模式初始化完成 (耗时: {load_time:.2f}秒)")
            except Exception as e:
                logger.error(f"规则模式初始化也失败: {e}")
                _initialization_status = "failed"
                extractor = None

        except Exception as e:
            logger.error(f"BERT 初始化失败: {e}", exc_info=True)
            # 尝试规则模式作为后备
            try:
                logger.info("尝试规则模式作为后备...")
                extractor = SmartNewsExtractor(use_bert=False, preload_db="property_translations.db")
                _initialization_status = "ready_fallback"
                load_time = time.time() - _initialization_start_time
                logger.info(f"规则模式初始化完成 (耗时: {load_time:.2f}秒)")
            except Exception as fallback_error:
                logger.error(f"规则模式初始化也失败: {fallback_error}")
                _initialization_status = "failed"
                extractor = None
        
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
