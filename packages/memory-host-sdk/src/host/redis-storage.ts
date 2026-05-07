import { Redis } from 'ioredis';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import { 
  MemorySearchManager, 
  MemorySearchResult, 
  MemoryReadResult, 
  MemoryProviderStatus, 
  MemorySyncProgressUpdate, 
  MemoryEmbeddingProbeResult, 
  MemorySearchRuntimeDebug 
} from './types.js';
import { createSubsystemLogger } from './openclaw-runtime-io.js';

const log = createSubsystemLogger("memory-redis");

export type RedisMemoryManagerOptions = {
  url?: string;
  ttlSeconds?: number;
  dbKey: string;
  localDbPath: string;
  innerManagerFactory: () => Promise<MemorySearchManager | null>;
};

/**
 * A MemorySearchManager adapter that persists a local SQLite-based memory store to Redis.
 * This is particularly useful for environments with ephemeral filesystems like Heroku.
 */
export class RedisMemoryManager implements MemorySearchManager {
  private redis: Redis | null = null;
  private inner: MemorySearchManager | null = null;
  private initializing: Promise<void> | null = null;

  constructor(private readonly options: RedisMemoryManagerOptions) {}

  private async ensureInitialized(): Promise<void> {
    if (this.inner) return;
    if (this.initializing) return this.initializing;

    this.initializing = (async () => {
      const url = this.options.url || process.env.REDIS_URL;
      if (!url) {
        throw new Error("Redis URL is required for Redis memory backend. Set REDIS_URL environment variable or memory.redis.url in openclaw.json.");
      }

      log.info(`Initializing Redis memory adapter (key: ${this.options.dbKey})`);

      try {
        this.redis = new Redis(url, {
          maxRetriesPerRequest: 3,
          retryStrategy: (times) => Math.min(times * 50, 2000),
          // Heroku Redis often requires TLS, but ioredis handles it via the URL if it starts with rediss://
          // Some Redis providers use self-signed certs.
          tls: url.startsWith('rediss://') ? { rejectUnauthorized: false } : undefined,
        });

        this.redis.on('error', (err) => {
          log.warn(`Redis connection error: ${String(err)}`);
        });

        // 1. Restore from Redis if available
        try {
          const data = await this.redis.getBuffer(this.options.dbKey);
          if (data && data.length > 0) {
            log.info(`Restoring memory from Redis (${this.options.dbKey}, ${data.length} bytes)`);
            await fs.mkdir(path.dirname(this.options.localDbPath), { recursive: true });
            await fs.writeFile(this.options.localDbPath, data);
          } else {
            log.info(`No existing memory found in Redis for key ${this.options.dbKey}`);
          }
        } catch (err) {
          log.warn(`Failed to restore memory from Redis: ${String(err)}`);
        }
      } catch (err) {
        log.error(`Failed to connect to Redis: ${String(err)}`);
        // We continue anyway, as the inner manager might still work locally (though data will be lost on restart)
      }

      // 2. Create inner manager
      this.inner = await this.options.innerManagerFactory();
      if (!this.inner) {
        throw new Error("Failed to create inner memory manager for Redis backend adapter.");
      }
    })();

    return this.initializing;
  }

  async search(
    query: string, 
    opts?: {
      maxResults?: number;
      minScore?: number;
      sessionKey?: string;
      qmdSearchModeOverride?: "query" | "search" | "vsearch";
      onDebug?: (debug: MemorySearchRuntimeDebug) => void;
      sources?: any[];
    }
  ): Promise<MemorySearchResult[]> {
    await this.ensureInitialized();
    opts?.onDebug?.({ backend: "redis" });
    return this.inner!.search(query, opts);
  }

  async readFile(params: { relPath: string; from?: number; lines?: number }): Promise<MemoryReadResult> {
    await this.ensureInitialized();
    return this.inner!.readFile(params);
  }

  status(): MemoryProviderStatus {
    if (!this.inner) {
      return {
        backend: "redis",
        provider: "uninitialized",
        sources: [],
      } as any;
    }
    const innerStatus = this.inner.status();
    return {
      ...innerStatus,
      backend: "redis",
      custom: {
        ...innerStatus.custom,
        redis: {
          connected: this.redis?.status === 'ready',
          key: this.options.dbKey,
        }
      }
    };
  }

  async sync(params?: {
    reason?: string;
    force?: boolean;
    sessionFiles?: string[];
    progress?: (update: MemorySyncProgressUpdate) => void;
  }): Promise<void> {
    await this.ensureInitialized();
    await this.inner!.sync?.(params);
    // Persistence after sync ensures LTM is updated in Redis
    await this.persistToRedis();
  }

  async probeEmbeddingAvailability(): Promise<MemoryEmbeddingProbeResult> {
    await this.ensureInitialized();
    return this.inner!.probeEmbeddingAvailability();
  }

  getCachedEmbeddingAvailability(): MemoryEmbeddingProbeResult | null {
    return this.inner?.getCachedEmbeddingAvailability?.() ?? null;
  }

  async probeVectorStoreAvailability(): Promise<boolean> {
    await this.ensureInitialized();
    return (await this.inner!.probeVectorStoreAvailability?.()) ?? this.inner!.probeVectorAvailability();
  }

  async probeVectorAvailability(): Promise<boolean> {
    await this.ensureInitialized();
    return this.inner!.probeVectorAvailability();
  }

  private async persistToRedis(): Promise<void> {
    if (!this.redis || this.redis.status !== 'ready' || !this.inner) return;

    try {
      if (fsSync.existsSync(this.options.localDbPath)) {
        // We read the file and push it to Redis. 
        // For very large databases this might be slow, but for OpenClaw LTM it's usually manageable.
        const data = await fs.readFile(this.options.localDbPath);
        if (data.length > 0) {
          log.info(`Persisting memory to Redis (${this.options.dbKey}, ${data.length} bytes)`);
          if (this.options.ttlSeconds) {
            await this.redis.setex(this.options.dbKey, this.options.ttlSeconds, data);
          } else {
            await this.redis.set(this.options.dbKey, data);
          }
        }
      }
    } catch (err) {
      log.warn(`Failed to persist memory to Redis: ${String(err)}`);
    }
  }

  async close(): Promise<void> {
    log.info(`Closing Redis memory adapter and performing final persistence...`);
    await this.persistToRedis();
    await this.inner?.close?.();
    this.redis?.disconnect();
  }
}
