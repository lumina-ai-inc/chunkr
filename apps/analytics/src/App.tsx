"use client"

import { useEffect, useState, useCallback } from 'react'
import { format } from "date-fns"
import { Calendar as CalendarIcon, ArrowLeft, ArrowRight } from "lucide-react"
import {
  XAxis, YAxis, CartesianGrid,
  BarChart, Bar, Legend
} from 'recharts'
import { ChartContainer, ChartTooltip } from './components/ui/chart'
import { Button } from './components/ui/button'
import { Input } from './components/ui/input'
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card'
import { Calendar } from "./components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "./components/ui/popover"
import { cn } from "@/lib/utils"
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell
} from './components/ui/table'
import { apiCall } from './api'

type PageData = {
  day: string
  pages: number
}

type StatusData = {
  day: string
  status: string
  pages: number
}

type UserData = {
  email: string
  total_pages: number
}

type TaskData = {
  task_id: string
  user_id: string
  email: string
  name: string
  page_count: number
  created_at: string
  completed_at: string | null
  status: string
  time_taken?: string
}

type StatusCounts = {
  Starting: number;
  Processing: number;
  Succeeded: number;
  Failed: number;
  Canceled: number;
  day: string;
}

type PaginatedTaskDetails = {
  tasks: TaskData[]
  page: number
  per_page: number
}

const calendarClassName = "bg-white text-gray-500 [&_.rdp-button:hover:not(.rdp-day_selected)]:bg-gray-200 [&_.rdp-day_selected]:bg-blue-600 [&_.rdp-day_selected]:hover:bg-blue-200 [&_.rdp-button]:hover:bg-gray-100 [&_.rdp-nav_button]:hover:bg-gray-100 [&_.rdp-head_cell]:font-medium [&_.rdp-day_today]:bg-gray-50 [&_.rdp-day_today]:font-normal [&_.rdp-button]:text-gray-600 [&_.rdp-day]:text-gray-600 [&_.rdp-day_selected]:text-white"

function AppWrapper() {
  return (
    <App />
  )
}

function App() {
  const [apiKey, setApiKey] = useState<string>('')
  const [isValidApiKey, setIsValidApiKey] = useState<boolean>(false)
  const [pagesData, setPagesData] = useState<PageData[]>([])
  const [statusData, setStatusData] = useState<StatusCounts[]>([])
  const [topUsers, setTopUsers] = useState<UserData[]>([])
  const [userEmail, setUserEmail] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [startDate, setStartDate] = useState<Date>(() => {
    const date = new Date()
    date.setMonth(date.getMonth() - 1)
    return date
  })
  const [endDate, setEndDate] = useState<Date>(() => new Date())
  const [taskData, setTaskData] = useState<TaskData[]>([])
  const [lifetimePages, setLifetimePages] = useState<number>(0)
  const [lastFetchTime, setLastFetchTime] = useState<number>(0)
  const FETCH_COOLDOWN = 120000
  const TASKS_PER_PAGE = 50
  const [taskPage, setTaskPage] = useState<number>(1)

  const handlePreviousPage = () => {
    setLastFetchTime(0);
    setTaskPage(p => Math.max(1, p - 1));
  };

  const handleNextPage = () => {
    setLastFetchTime(0);
    setTaskPage(p => p + 1);
  };

  const handleStartDateMonthChange = (month: Date) => {
    const originalDay = startDate.getDate();
    let newStartDate = new Date(month.getFullYear(), month.getMonth(), originalDay);

    if (newStartDate.getMonth() !== month.getMonth()) {
      newStartDate = new Date(month.getFullYear(), month.getMonth() + 1, 0);
    }
    setStartDate(newStartDate);
    setLastFetchTime(0);
  };

  const handleEndDateMonthChange = (month: Date) => {
    const originalDay = endDate.getDate();
    let newEndDate = new Date(month.getFullYear(), month.getMonth(), originalDay);

    if (newEndDate.getMonth() !== month.getMonth()) {
      newEndDate = new Date(month.getFullYear(), month.getMonth() + 1, 0);
    }
    setEndDate(newEndDate);
    setLastFetchTime(0);
  };

  const fetchData = useCallback(async () => {
    const now = Date.now()
    if (!isValidApiKey || !startDate || !endDate || !apiKey || isLoading ||
      (now - lastFetchTime < FETCH_COOLDOWN && lastFetchTime !== 0)) {
      return
    }
    setIsLoading(true)
    setLastFetchTime(now)

    try {
      const start = startDate.toISOString()
      const end = endDate.toISOString()

      const params = new URLSearchParams({
        start,
        end,
        page: taskPage.toString(),
        per_page: TASKS_PER_PAGE.toString(),
      })
      if (userEmail) params.append('email', userEmail)

      const headers = { 'Authorization': `Bearer ${apiKey}` }

      const [pData, sData, tuData, tDataRes, ltPages] = await Promise.all([
        apiCall<PageData[]>(`/pages-per-day?${params.toString()}`, { headers }),
        apiCall<StatusData[]>(`/status-breakdown?${params.toString()}`, { headers }),
        apiCall<UserData[]>('/top-users', {
          method: 'POST',
          body: { start, end, limit: 5 },
          headers
        }),
        apiCall<PaginatedTaskDetails>(`/task-details?${params.toString()}`, { headers }),
        apiCall<number>('/lifetime-pages', { headers })
      ])

      const filledPagesData = fillMissingPages(pData, startDate, endDate)
      const filledStatusData = fillMissingStatus(sData, startDate, endDate)
      setPagesData(filledPagesData)
      setStatusData(filledStatusData)
      setTopUsers(tuData)
      setTaskData(tDataRes.tasks)
      setLifetimePages(ltPages)
    } catch (err: any) {
      console.error('Failed:', err)
      if (err.status === 401 || err.status === 403) {
        console.error("API Key validation failed or expired.")
        logout()
      }
    } finally {
      setIsLoading(false)
    }
  }, [startDate, endDate, userEmail, apiKey, isValidApiKey, isLoading, lastFetchTime, taskPage, FETCH_COOLDOWN])

  useEffect(() => {
    if (startDate && endDate && isValidApiKey) {
      fetchData()

      const interval = setInterval(() => {
        fetchData()
      }, FETCH_COOLDOWN)

      return () => clearInterval(interval)
    }
  }, [startDate, endDate, userEmail, fetchData, isValidApiKey])

  const handleRefresh = () => {
    setLastFetchTime(0)
    fetchData()
  }

  const validateApiKey = async (key: string) => {
    if (!key) return
    try {
      await apiCall<number>('/lifetime-pages', {
        headers: { Authorization: `Bearer ${key}` }
      })
      localStorage.setItem('apiKey', key)
      setIsValidApiKey(true)
    } catch {
      localStorage.removeItem('apiKey')
      setIsValidApiKey(false)
      alert("Invalid API Key provided.")
    }
  }

  useEffect(() => {
    const savedApiKey = localStorage.getItem('apiKey')
    if (savedApiKey) {
      setApiKey(savedApiKey)
      validateApiKey(savedApiKey)
    }
  }, [])

  const totalPagesInDateRange = pagesData.reduce((sum, item) => sum + item.pages, 0)

  const todayStr = new Date().toISOString().split('T')[0]
  const pagesToday = pagesData.find(item => item.day === todayStr)?.pages || 0

  const currentMonth = new Date().getMonth()
  const currentYear = new Date().getFullYear()
  const pagesThisMonth = pagesData.reduce((sum, item) => {
    const itemDate = new Date(item.day)
    return itemDate.getMonth() === currentMonth &&
      itemDate.getFullYear() === currentYear ?
      sum + item.pages : sum
  }, 0)

  const searchUser = (email: string) => {
    setLastFetchTime(0);
    setUserEmail(email)
  }

  const clearSearch = () => {
    setLastFetchTime(0);
    setUserEmail('')
  }

  const fillMissingPages = (data: PageData[], start: Date, end: Date): PageData[] => {
    const map = new Map<string, number>()
    const cursor = new Date(start)
    while (cursor <= end) {
      const key = cursor.toISOString().split('T')[0]
      map.set(key, 0)
      cursor.setDate(cursor.getDate() + 1)
    }
    data.forEach(item => {
      map.set(item.day, (map.get(item.day) || 0) + item.pages)
    })
    return Array.from(map, ([day, pages]) => ({ day, pages }))
  }

  const fillMissingStatus = (data: StatusData[], start: Date, end: Date): StatusCounts[] => {
    const map = new Map<string, Omit<StatusCounts, 'day'>>()
    const cursor = new Date(start)

    while (cursor <= end) {
      const day = cursor.toISOString().split('T')[0]
      map.set(day, {
        Starting: 0,
        Processing: 0,
        Succeeded: 0,
        Failed: 0,
        Canceled: 0
      })
      cursor.setDate(cursor.getDate() + 1)
    }

    data.forEach(item => {
      const dayData = map.get(item.day)
      if (dayData) {
        dayData[item.status as keyof Omit<StatusCounts, 'day'>] = item.pages
      }
    })

    return Array.from(map.entries()).map(([day, counts]) => ({
      day,
      ...counts
    }))
  }

  const getStatusStyle = (status: string) => {
    switch (status.toLowerCase()) {
      case 'starting':
        return "bg-yellow-50 text-yellow-700"
      case 'processing':
        return "bg-blue-50 text-blue-700"
      case 'succeeded':
        return "bg-green-50 text-green-700"
      case 'failed':
        return "bg-red-50 text-red-700"
      case 'canceled':
        return "bg-gray-50 text-gray-700"
      default:
        return "bg-gray-50 text-gray-700"
    }
  }

  const shiftStartDateLeft = async () => {
    setLastFetchTime(0);
    setStartDate(prev => shiftDate(prev, -1));
  }

  const shiftStartDateRight = async () => {
    setLastFetchTime(0);
    setStartDate(prev => shiftDate(prev, 1));
  }

  const shiftEndDateLeft = async () => {
    setLastFetchTime(0);
    setEndDate(prev => shiftDate(prev, -1));
  }

  const shiftEndDateRight = async () => {
    setLastFetchTime(0);
    setEndDate(prev => shiftDate(prev, 1));
  }

  const shiftDate = (date: Date, days: number) => {
    const newDate = new Date(date)
    newDate.setDate(newDate.getDate() + days)
    return newDate
  }

  const getTimeTaken = (created: string, completed: string | null) => {
    if (!completed) return '-'
    const start = new Date(created)
    const end = new Date(completed)
    const diff = (end.getTime() - start.getTime()) / 1000

    if (diff < 0) return 'N/A'
    return `${diff}s`
  }

  const getPagesPerSecond = (pageCount: number, created: string, completed: string | null) => {
    if (!completed) return '-'
    const start = new Date(created)
    const end = new Date(completed)
    const diff = (end.getTime() - start.getTime()) / 1000

    if (diff <= 0) return 'N/A'
    return (pageCount / diff).toFixed(2)
  }

  const logout = () => {
    localStorage.removeItem('apiKey')
    setIsValidApiKey(false)
    setApiKey('')
  }

  if (!isValidApiKey) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-100">
        <Card className="w-full max-w-md shadow-lg">
          <CardHeader>
            <CardTitle className="text-center text-2xl font-semibold">Enter API Key</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input
              type="password"
              placeholder="API Key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="border-gray-300 focus:ring-blue-500 focus:border-blue-500"
            />
            <Button onClick={() => validateApiKey(apiKey)} className="w-full bg-blue-600 hover:bg-blue-700 text-white">
              Validate & Enter
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="p-4 md:p-8 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold text-center">Analytics Dashboard</h1>

      <div className="flex justify-between mb-4">
        <Button variant="outline" onClick={handleRefresh} disabled={isLoading}>
          {isLoading ? "Loading..." : "Refresh Data"}
        </Button>
        <Button variant="outline" onClick={logout}>Log Out</Button>
      </div>

      <div className="flex flex-col items-center gap-4 mb-6">
        <div className="flex gap-4">
          <div className="flex flex-col items-center gap-2">
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "w-[240px] justify-start text-left font-normal",
                    !startDate && "text-muted-foreground"
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {startDate ? format(startDate, "PPP") : <span>Start date</span>}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0 bg-white">
                <Calendar
                  mode="single"
                  selected={startDate}
                  onSelect={(date) => { setStartDate(date || new Date()); setLastFetchTime(0); }}
                  onMonthChange={handleStartDateMonthChange}
                  initialFocus
                  toDate={endDate}
                  className={calendarClassName}
                />
              </PopoverContent>
            </Popover>
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={shiftStartDateLeft}
                className="bg-gray-100 hover:bg-gray-200"
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                onClick={shiftStartDateRight}
                className="bg-gray-100 hover:bg-gray-200"
              >
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <div className="flex flex-col items-center gap-2">
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "w-[240px] justify-start text-left font-normal",
                    !endDate && "text-muted-foreground"
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {endDate ? format(endDate, "PPP") : <span>End date</span>}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0 bg-white">
                <Calendar
                  mode="single"
                  selected={endDate}
                  onSelect={(date) => { setEndDate(date || new Date()); setLastFetchTime(0); }}
                  onMonthChange={handleEndDateMonthChange}
                  initialFocus
                  toDate={new Date()}
                  fromDate={startDate}
                  className={calendarClassName}
                />
              </PopoverContent>
            </Popover>
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={shiftEndDateLeft}
                className="bg-gray-100 hover:bg-gray-200"
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                onClick={shiftEndDateRight}
                className="bg-gray-100 hover:bg-gray-200"
              >
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>

        <div className="flex gap-4 mb-4">
          <Input
            type="email"
            placeholder="Enter email"
            value={userEmail}
            onChange={(e) => {
              setUserEmail(e.target.value);
              setLastFetchTime(0);
            }}
          />
          <Button onClick={() => searchUser(userEmail)}>Search</Button>
          {userEmail && (
            <Button variant="outline" onClick={clearSearch}>Clear</Button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Lifetime Pages</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{lifetimePages.toLocaleString()}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Total Pages This Date Range</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{totalPagesInDateRange.toLocaleString()}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Pages Processed Today</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{pagesToday.toLocaleString()}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Pages This Month</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{pagesThisMonth.toLocaleString()}</div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Pages Processed Per Day</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer config={{}} className="h-[300px]">
                <BarChart data={pagesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <ChartTooltip />
                  <Bar dataKey="pages" fill="hsl(var(--chart-1))" name="Pages" />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Task Status Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer config={{}} className="h-[300px]">
                <BarChart data={statusData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <ChartTooltip />
                  <Legend />
                  <Bar stackId="status" dataKey="Starting" fill="#FFB020" />
                  <Bar stackId="status" dataKey="Processing" fill="#3E7BFA" />
                  <Bar stackId="status" dataKey="Succeeded" fill="#10B981" />
                  <Bar stackId="status" dataKey="Failed" fill="#EF4444" />
                  <Bar stackId="status" dataKey="Canceled" fill="#6B7280" />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </div>

        {(!userEmail && topUsers.length >= 5) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Top Users</CardTitle>
              </CardHeader>
              <CardContent>
                <ChartContainer config={{}} className="h-[300px]">
                  <BarChart layout="vertical" data={topUsers}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="email" width={150} />
                    <ChartTooltip />
                    <Bar dataKey="total_pages" fill="hsl(var(--chart-1))" name="Pages" />
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Tasks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Task ID</TableHead>
                    <TableHead>User</TableHead>
                    <TableHead>Pages</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Completed</TableHead>
                    <TableHead>Time Taken</TableHead>
                    <TableHead>Pages per Sec</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {taskData.map((task) => (
                    <TableRow key={task.task_id}>
                      <TableCell className="font-mono">{task.task_id}</TableCell>
                      <TableCell>
                        <div className="flex flex-col">
                          <span>{task.name}</span>
                          <span className="text-sm text-muted-foreground">{task.email}</span>
                        </div>
                      </TableCell>
                      <TableCell>{task.page_count}</TableCell>
                      <TableCell>
                        <span className={cn(
                          "inline-flex items-center rounded-full px-2 py-1 text-xs font-medium",
                          getStatusStyle(task.status)
                        )}>
                          {task.status}
                        </span>
                      </TableCell>
                      <TableCell>{new Date(task.created_at).toLocaleString()}</TableCell>
                      <TableCell>
                        {task.completed_at ? new Date(task.completed_at).toLocaleString() : '-'}
                      </TableCell>
                      <TableCell>
                        {getTimeTaken(task.created_at, task.completed_at)}
                      </TableCell>
                      <TableCell>
                        {getPagesPerSecond(task.page_count, task.created_at, task.completed_at)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            <div className="ml-auto flex gap-2">
              <Button variant="outline" disabled={taskPage === 1} onClick={handlePreviousPage}>
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <Button variant="outline" disabled={taskData.length < TASKS_PER_PAGE} onClick={handleNextPage}>
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default AppWrapper
