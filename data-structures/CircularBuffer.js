import { logger } from '../utils.js';

export class CircularBuffer {
    constructor(capacity) {
        this.capacity = Math.max(1, Math.floor(Number.isFinite(capacity) ? capacity : 10));
        this.buffer = new Array(this.capacity).fill(null);
        this.start = 0;
        this.size = 0;
    }

    push(item) {
        if (item === null || item === undefined) {
            logger.warn('CircularBuffer.push: Invalid item (null/undefined). Skipping.', { item });
            return;
        }
        const index = (this.start + this.size) % this.capacity;
        this.buffer[index] = item;
        if (this.size < this.capacity) {
            this.size++;
        } else {
            this.start = (this.start + 1) % this.capacity;
        }
    }

    get(index) {
        if (index < 0 || index >= this.size) return null;
        return this.buffer[(this.start + index) % this.capacity];
    }

    getAll() {
        const result = [];
        for (let i = 0; i < this.size; i++) {
            const item = this.get(i);
            if (item !== null) {
                result.push(item);
            }
        }
        return result;
    }
    
    get length() {
        return this.size;
    }

    clear() {
        this.buffer.fill(null);
        this.start = 0;
        this.size = 0;
    }
}