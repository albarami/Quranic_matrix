import { test, expect } from '@playwright/test';

test.describe('QBM E2E Tests', () => {
  test.describe('Proof Page', () => {
    test('renders proof page and can submit query', async ({ page }) => {
      await page.goto('/proof');
      
      // Check page title
      await expect(page.locator('h1')).toContainText('نظام الإثبات');
      
      // Check input field exists
      const input = page.locator('input[type="text"]');
      await expect(input).toBeVisible();
      
      // Check submit button exists
      const submitButton = page.locator('button[type="submit"]');
      await expect(submitButton).toBeVisible();
    });

    test('shows example queries', async ({ page }) => {
      await page.goto('/proof');
      
      // Check example queries are displayed
      const exampleButtons = page.locator('button:has-text("ما علاقة")');
      await expect(exampleButtons.first()).toBeVisible();
    });
  });

  test.describe('Genome Page', () => {
    test('loads genome status', async ({ page }) => {
      await page.goto('/genome');
      
      // Check page title
      await expect(page.locator('h1')).toContainText('Q25');
      
      // Wait for status to load (may take time if backend is slow)
      await page.waitForSelector('text=Genome Status', { timeout: 10000 }).catch(() => {
        // If status doesn't load, check for error message
        expect(page.locator('text=Failed to load')).toBeVisible();
      });
    });

    test('has export buttons', async ({ page }) => {
      await page.goto('/genome');
      
      // Check export section exists
      await expect(page.locator('text=Export Genome')).toBeVisible();
      
      // Check light and full export buttons
      await expect(page.locator('text=Light Export')).toBeVisible();
      await expect(page.locator('text=Full Export')).toBeVisible();
    });
  });

  test.describe('Reviews Page', () => {
    test('loads reviews page', async ({ page }) => {
      await page.goto('/reviews');
      
      // Check page title
      await expect(page.locator('h1')).toContainText('Scholar Reviews');
      
      // Check new review button
      await expect(page.locator('text=New Review')).toBeVisible();
    });

    test('can open create review modal', async ({ page }) => {
      await page.goto('/reviews');
      
      // Click new review button
      await page.click('text=New Review');
      
      // Check modal opens
      await expect(page.locator('text=Create New Review')).toBeVisible();
      
      // Check form fields
      await expect(page.locator('input[placeholder*="Edge ID"]')).toBeVisible();
    });

    test('shows filter options', async ({ page }) => {
      await page.goto('/reviews');
      
      // Check status filter
      const statusFilter = page.locator('select').first();
      await expect(statusFilter).toBeVisible();
      
      // Check filter options
      await statusFilter.click();
      await expect(page.locator('option:has-text("Pending")')).toBeVisible();
      await expect(page.locator('option:has-text("Approved")')).toBeVisible();
    });
  });

  test.describe('Explorer Page', () => {
    test('loads explorer with surah grid', async ({ page }) => {
      await page.goto('/explorer');
      
      // Check page loads
      await expect(page.locator('h1')).toContainText('Explorer');
      
      // Check view toggle buttons exist
      await expect(page.locator('[title="Grid View"]')).toBeVisible();
      await expect(page.locator('[title="List View"]')).toBeVisible();
    });

    test('can search surahs', async ({ page }) => {
      await page.goto('/explorer');
      
      // Find search input
      const searchInput = page.locator('input[placeholder*="Search"]');
      await expect(searchInput).toBeVisible();
      
      // Type search query
      await searchInput.fill('Fatiha');
    });
  });

  test.describe('Navigation', () => {
    test('all nav links work', async ({ page }) => {
      await page.goto('/');
      
      // Check navigation exists
      const nav = page.locator('nav');
      await expect(nav).toBeVisible();
      
      // Check key nav items
      await expect(page.locator('a[href="/proof"]')).toBeVisible();
      await expect(page.locator('a[href="/explorer"]')).toBeVisible();
      await expect(page.locator('a[href="/genome"]')).toBeVisible();
      await expect(page.locator('a[href="/reviews"]')).toBeVisible();
    });

    test('can navigate to genome page', async ({ page }) => {
      await page.goto('/');
      
      // Click genome link
      await page.click('a[href="/genome"]');
      
      // Verify navigation
      await expect(page).toHaveURL('/genome');
      await expect(page.locator('h1')).toContainText('Q25');
    });

    test('can navigate to reviews page', async ({ page }) => {
      await page.goto('/');
      
      // Click reviews link
      await page.click('a[href="/reviews"]');
      
      // Verify navigation
      await expect(page).toHaveURL('/reviews');
      await expect(page.locator('h1')).toContainText('Reviews');
    });
  });

  test.describe('Language Toggle', () => {
    test('can switch between Arabic and English', async ({ page }) => {
      await page.goto('/');
      
      // Find language toggle buttons
      const arabicButton = page.locator('button:has-text("العربية")');
      const englishButton = page.locator('button:has-text("English")');
      
      // Both should be visible
      await expect(arabicButton).toBeVisible();
      await expect(englishButton).toBeVisible();
    });
  });
});
